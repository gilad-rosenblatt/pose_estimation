import math
from abc import ABC, abstractmethod

import cv2
import numpy as np
import tensorflow as tf

from encoders import DetectionsEncoder, KeypointsEncoder
from parsers import DetectionsParser, KeypointsParser
from plotters import DetectionsPlotter, KeypointsPlotter


class Dataset(tf.keras.utils.Sequence, ABC):
    """Generator of image batches and their ground truth annotations."""

    def __init__(self, batch_size=64, dataset="train", shuffle=False):
        """
        Load dataset annotations and save batch size and configuration flags.

        :param int batch_size: number of images to include in a single batch.
        :param str dataset: the dataset to load (either "train" or "validation").
        :param bool shuffle: if True shuffle images after each epoch.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._parser = self._get_parser(dataset=dataset)
        self._encoder = self._get_encoder()
        self._indices = np.arange(0, len(self._parser.info))

    def __len__(self):
        """
        :return int: the number of batches in the dataset (batches per epoch).
        """
        return math.ceil(len(self._parser.info) / self.batch_size)

    def __getitem__(self, batch_index):
        """
        Generate one batch of data.

        :param int batch_index: index of the batch to get.
        :return list: batch info as a list of length batch_size where each item contains a tuple for one annotation.
        """
        indices = self._indices[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        return [self._parser.info[index] for index in indices]  # Using indices proxy allows shuffling at epoch end.

    def on_epoch_end(self):
        """Shuffle dataset on epoch end if shuffle flag is True."""
        if self.shuffle:
            np.random.shuffle(self._indices)

    @classmethod
    @abstractmethod
    def _get_parser(cls, dataset):
        pass

    @classmethod
    @abstractmethod
    def _get_encoder(cls):
        pass


class DetectionsDataset(Dataset):
    """Generator of image batches and their ground truth bounding boxes for single-class one-step object detection."""

    # Input image shape without channels (height, width).
    INPUT_SHAPE = (192, 192)

    # Output cells shape without channels (height, width).
    OUTPUT_SHAPE = (6, 6)

    def __init__(self, generate_image_ids=False, *args, **kwargs):
        """
        Load dataset detection annotations and save configuration flags.

        :param generate_image_ids: if True generates image IDs (x, y, image_id) with each batch (for scoring purposes).
        """
        super().__init__(*args, **kwargs)
        self.generate_image_ids = generate_image_ids

    def __getitem__(self, batch_index):
        """
        Generate one batch of data corresponding to the input batch index. Does not raise IndexError (a tf.Sequence).

        :param int batch_index: index of the batch to get.
        :return tuple: x of shape (batch_size, *INPUT_SHAPE, 3) and y of shape (batch_size, *OUTPUT_SHAPE, 5).
        """

        # Get file paths and box annotations for this batch.
        batch_info = super().__getitem__(batch_index=batch_index)

        # Initialize input and output arrays (take batch size from indices since last batch can be smaller).
        x = np.empty(shape=(len(batch_info), *DetectionsDataset.INPUT_SHAPE, 3), dtype=np.float32)
        y = np.empty(shape=(len(batch_info), *DetectionsDataset.OUTPUT_SHAPE, 5), dtype=np.float32)
        ids = np.empty(shape=(len(batch_info),), dtype=int) if self.generate_image_ids else None

        # Fill input and output arrays image-by-image. Does not raise IndexError (last batch check done by tf.Sequence).
        for index, (filename, box_list, image_id) in enumerate(batch_info):
            # Load the image and instantiate a numpy array of boxes (one per row).
            this_image = cv2.imread(self._parser.get_path(filename))
            boxes = np.array(box_list, dtype=np.float32)

            # Resize image to input shape and rescale boxes accordingly.
            this_image, boxes = self._resize(this_image, boxes)

            # Normalize input image to 0-1.
            this_x = this_image / 255

            # Encode boxes into a cell grid output array.
            this_y = self._encoder.encode(boxes)

            # Append this x and y to the batch (and add this image ID).
            x[index, ...] = this_x
            y[index, ...] = this_y
            if self.generate_image_ids:
                ids[index] = image_id

        # Return batch.
        return (x, y) if not self.generate_image_ids else (x, y, ids)

    @classmethod
    def _get_parser(cls, dataset):
        return DetectionsParser(dataset=dataset)

    @classmethod
    def _get_encoder(cls):
        return DetectionsEncoder(
            input_shape=DetectionsDataset.INPUT_SHAPE,
            output_shape=DetectionsDataset.OUTPUT_SHAPE
        )

    @staticmethod
    def _resize(image, boxes):
        """
        Resize input image and associated bounding boxes to fit the model input shape.

        :param np.ndarray image: image to resize.
        :param np.ndarray boxes: (num_boxes, 4) bounding boxes in row x1, y1, w, h format (1: upper left corner).
        :return tuple: image resized to input shape and bounding boxes resized accordingly.
        """
        height, width, _ = image.shape
        boxes = DetectionsEncoder.scale_boxes(
            boxes=boxes,
            from_shape=(height, width),
            to_shape=DetectionsDataset.INPUT_SHAPE
        )
        image = cv2.resize(image, dsize=tuple(reversed(DetectionsDataset.INPUT_SHAPE)))  # Use opencv convention (w, h).
        return image, boxes

    def show(self, resize=False):
        """
        Look at the images and corresponding bounding boxes of the entire dataset one by one (press 'q' to exit).

        :param bool resize: if True resize images and bounding boxes to INPUT_SHAPE (otherwise use original shape).
        """

        # Creates generator that loads each image in sequence and instantiate a numpy array for its boxes.
        generator = (
            (
                cv2.imread(self._parser.get_path(filename)),
                np.array(box_list, dtype=np.float32)
            )
            for filename, box_list, _ in self._parser.info
        )

        # Resize image and boxes to model input shape.
        if resize:
            generator = (self._resize(image, boxes) for image, boxes in generator)

        # Show the image and bounding boxes.
        DetectionsPlotter.show_generator(generator)


class KeypointsDataset(Dataset):
    """Generator of image batches and their ground truth keypoints encoded as heatmaps for pose estimation."""

    # Input image shape without channels (height, width).
    INPUT_SHAPE = (256, 192)

    # Output heatmap shape without channels (height, width).
    OUTPUT_SHAPE = (64, 48)

    def __getitem__(self, batch_index):
        """
        Generate one batch of data corresponding to the input batch index. Does not raise IndexError (a tf.Sequence).

        :param int batch_index: index of the batch to get.
        :return tuple: x of shape (batch_size, *INPUT_SHAPE, 3) and y of shape (batch_size, *OUTPUT_SHAPE, 17).
        """
        # Get file paths and keypoint annotations for this batch.
        batch_info = super().__getitem__(batch_index=batch_index)

        # Initialize input and output arrays (take batch size from indices since last batch can be smaller).
        x = np.empty(shape=(len(batch_info), *KeypointsDataset.INPUT_SHAPE, 3), dtype=np.float32)
        y = np.empty(shape=(len(batch_info), *KeypointsDataset.OUTPUT_SHAPE, 17), dtype=np.float32)

        # Fill input and output arrays image-by-image. Does not raise IndexError (last batch check done by tf.Sequence).
        for index, (filename, box, keypoints) in enumerate(batch_info):

            # Load the image and instantiate numpy arrays for the detection box and keypoints (one per row).
            this_image = cv2.imread(self._parser.get_path(filename))
            box = np.array(box, dtype=np.float32)
            keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)

            # Crop and resize image to input shape and move/rescale keypoints to input coordinates.
            this_image, boxes = self._resize(image=this_image, box=box, keypoints=keypoints)

            # Normalize input image to 0-1.
            this_x = this_image / 255

            # Encode boxes into a heatmap in output dimensions.
            this_y = self._encoder.encode(keypoints)

            # Append this x and y to the batch.
            x[index, ...] = this_x
            y[index, ...] = this_y

        # Return batch.
        return x, y

    @classmethod
    def _get_parser(cls, dataset):
        return KeypointsParser(dataset=dataset)

    @classmethod
    def _get_encoder(cls):
        return KeypointsEncoder(
            input_shape=cls.INPUT_SHAPE,
            output_shape=cls.OUTPUT_SHAPE
        )

    @classmethod
    def _resize(cls, image, box, keypoints, return_box=False):
        """
        Crop input image to input shape guided by the detection bounding box and map keypoints to crop coordinates.

        :param np.ndarray image: image to resize.
        :param np.ndarray box: (4,) original detection bounding box in x1, y1, w, h format (1: upper left corner).
        :param np.ndarray keypoints: (num_keypoints, 3) keypoints array in x, y, visible format (0s for missing points).
        :param bool return_box: if True also returns the box in the input crop coordinates.
        :return tuple: image crop having input shape and keypoints in crop x, y coordinates.
        """

        # Expand the detection box to include all keypoints and fit the input shape aspect ratio (not guaranteed).
        input_height, input_width = cls.INPUT_SHAPE
        x1, y1, w, h = KeypointsEncoder.expand_box(
            box=box,
            keypoints=keypoints,
            image_shape=image.shape[:2],
            aspect_ratio=input_width / input_height
        )

        # Cut the crop and move keypoints to crop coordinates.
        crop = image[int(y1):int(y1 + h), int(x1):int(x1 + w), ...]
        moved_keypoints = KeypointsEncoder.move_keypoints(
            origin=np.array([int(x1), int(y1)], dtype=np.float32),
            keypoints=keypoints
        )

        # Scale keypoints and resize the crop to input shape (can distort in edge case).
        input_crop = cv2.resize(crop, dsize=tuple(reversed(cls.INPUT_SHAPE)))  # Use opencv convention (w, h).
        input_keypoints = KeypointsEncoder.scale_keypoints(
            keypoints=moved_keypoints,  # Keypoints in crop (expanded box) x, y.
            from_shape=(int(y1 + h) - int(y1), int(x1 + w) - int(x1)),  # Expanded box shape.
            to_shape=cls.INPUT_SHAPE
        )

        # Return image crop and associated keypoints in input shape/coordinates.
        return (input_crop, input_keypoints) if not return_box else (input_crop, None, input_keypoints)  # FIXME box.

    def show(self, resize=False):
        """
        Look at the images and corresponding keypoints of the entire dataset one by one (press 'q' to exit).

        :param bool resize: if True resize images and keypoints to INPUT_SHAPE (otherwise use original shape).
        """

        # Creates generator that loads each image in sequence and instantiate a numpy array for its boxes.
        generator = (
            (
                cv2.imread(self._parser.get_path(filename)),
                np.array(box, dtype=np.float32),
                np.array(keypoints, dtype=np.float32).reshape(-1, 3)
            )
            for filename, box, keypoints in self._parser.info
        )

        # Resize image and boxes to model input shape.
        if resize:
            generator = (self._resize(image, box, keypoints, return_box=True) for image, box, keypoints in generator)

        # Show the image bounding box and keypoints.
        KeypointsPlotter.show_generator(generator)


if __name__ == "__main__":
    """Run a few quick tests for dataset generators."""
    check_keypoints = True
    factory = KeypointsDataset if check_keypoints else DetectionsDataset
    ds = factory(dataset="validation", shuffle=True)
    ds.show(resize=False)  # Press 'q' to quit displaying images.
    print(f"Number of batches in validation dataset: {len(ds)}")
    for batch_number, (x, y) in enumerate(ds):
        print(f"batch {batch_number}: x.shape={x.shape}, y.shape={y.shape}")
