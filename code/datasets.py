import math

import cv2
import numpy as np
import tensorflow as tf

from encoders import DetectionsEncoder
from parsers import DetectionsParser
from plotters import DetectionsPlotter


class Dataset(tf.keras.utils.Sequence):
    """Generator of image batches and their ground truth bounding boxes for single-class one-step object detection."""

    # Input image shape without channels (height, width).
    IMAGE_SHAPE = (192, 192)

    # Output cells shape without channels (height, width).
    CELLS_SHAPE = (6, 6)

    def __init__(self, batch_size=64, dataset="train", shuffle=False, generate_image_ids=False):
        """
        Load dataset annotations and save batch size and configuration flags.

        :param int batch_size: number of images to include in a single batch.
        :param str dataset: the dataset to load (either "train" or "validation").
        :param bool shuffle: if True shuffle images after each epoch.
        :param generate_image_ids: if True generates image IDs (x, y, image_id) with each batch (gor scoring purposes).
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generate_image_ids = generate_image_ids
        self._parser = DetectionsParser(dataset=dataset)
        self._indices = np.arange(0, len(self._parser.info))
        self._encoder = DetectionsEncoder(input_shape=Dataset.IMAGE_SHAPE, output_shape=Dataset.CELLS_SHAPE)

    def __len__(self):
        """
        :return int: the number of batches in the dataset (batches per epoch).
        """
        return math.ceil(len(self._parser.info) / self.batch_size)

    def __getitem__(self, batch_index):
        """
        Generate one batch of data corresponding to the input batch index. Does not raise IndexError (a tf.Sequence).

        :param int batch_index: index of the batch to get.
        :return tuple: x of shape [batch_size, height, width, channels] and y of shape [batch_size, *CELLS_SHAPE, 5].
        """

        # Get file paths and box annotations for this batch (using the indices proxy allows shuffling at epoch end).
        indices = self._indices[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        batch_info = [self._parser.info[index] for index in indices]

        # Initialize input and output arrays (take batch size from indices since last batch can be smaller).
        x = np.empty(shape=(indices.size, *Dataset.IMAGE_SHAPE, 3), dtype=np.float32)
        y = np.empty(shape=(indices.size, *Dataset.CELLS_SHAPE, 5), dtype=np.float32)
        ids = np.empty(shape=(indices.size,), dtype=int) if self.generate_image_ids else None

        # Fill input and output arrays image-by-image. Does not raise IndexError (last batch check done by tf.Sequence).
        for index, (filename, box_list, image_id) in enumerate(batch_info):
            # Load the image and instantiate a numpy array of boxes (one per row).
            this_image = cv2.imread(self._parser.get_path(filename))
            boxes = np.array(box_list, dtype=np.float32)

            # Resize image to model input shape and rescale boxes accordingly.
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

    def on_epoch_end(self):
        """Shuffle dataset on epoch end if shuffle flag is True."""
        if self.shuffle:
            np.random.shuffle(self._indices)

    @staticmethod
    def _resize(image, boxes):
        """
        Resize input image and associated bounding boxes to fit the model input shape.

        :param np.ndarray image: image to resize.
        :param np.ndarray boxes: bounding boxes in the row format [x1, y1, w, h] (upper left corner, width and height).
        :return tuple: images resized to class resize shape (model input shape) and bounding boxes resized accordingly.
        """
        height, width, _ = image.shape
        boxes = DetectionsEncoder.scale_boxes(boxes=boxes, from_shape=(height, width), to_shape=Dataset.IMAGE_SHAPE)
        image = cv2.resize(image, dsize=tuple(reversed(Dataset.IMAGE_SHAPE)))  # Use opencv convention (width, height).
        return image, boxes

    def show(self, resize=False):
        """
        Look at the images and corresponding bounding boxes of the entire dataset one by one (press 'q' to exit).

        :param bool resize: if True resize images and bounding boxes to IMAGE_SHAPE (otherwise use original shape).
        """

        # Creates generator that loads each image in sequence and instantiate a numpy array for its boxes.
        generator = (
            (cv2.imread(self._parser.get_path(filename)), np.array(box_list, dtype=np.float32))
            for filename, box_list, _ in self._parser.info
        )

        # Resize image and boxes to model input shape.
        if resize:
            generator = (self._resize(image, boxes) for image, boxes in generator)

        # Show the image and boxes.
        DetectionsPlotter.show_generator(generator)


if __name__ == "__main__":
    """Run a few quick tests for the dataset generator."""
    ds = Dataset(dataset="validation", shuffle=True)
    ds.show(resize=False)  # Press 'q' to quit displaying images.
    print(f"Number of batches in validation dataset: {len(ds)}")
    for batch_number, (x, y) in enumerate(ds):
        print(f"batch {batch_number}: x.shape={x.shape}, y.shape={y.shape}")
