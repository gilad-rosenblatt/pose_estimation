import random
import math
import numpy as np
import cv2
import tensorflow as tf
from plotters import Plotter
from encoder import BoxEncoder
from parsers import Parser


# TODO add data augmentation.


class Dataset(tf.keras.utils.Sequence):
    """Generator of image batches and their ground truth bounding boxes for single-class one-step object detection."""

    # Input image shape without channels (height, width).
    IMAGE_SHAPE = (192, 192)

    # Output cells shape without channels (height, width).
    CELLS_SHAPE = (6, 6)

    def __init__(self, batch_size=64, dataset="train", shuffle=False, yield_y=True, output_type=2):
        """
        Load dataset annotations and save batch size and configuration flags.

        :param int batch_size: number of images to include in a single batch.
        :param str dataset: the dataset to load (either "train" or "validation").
        :param bool shuffle: if True shuffle images after each epoch.
        :param bool yield_y: if True generate only x without y (e.g., for predicting on x).
        :param int output_type: y output for each cell - 0 (detected y/n), 1 (y/n + x & y shift), 2 (y/n + box).
        """
        self.batch_size = batch_size
        self._parser = Parser(dataset=dataset)
        self.shuffle = shuffle
        self.yield_y = yield_y
        self.type = output_type
        self._encoder = BoxEncoder(image_shape=Dataset.IMAGE_SHAPE, cells_shape=Dataset.CELLS_SHAPE)

    def __len__(self):
        """:return int: the number of batches in the dataset (per epoch)."""
        return math.ceil(len(self._parser.info) / self.batch_size)

    def __getitem__(self, batch_index):
        """
        Generate one batch of data corresponding to the input batch index.

        :param int batch_index: index of the batch to get.
        :return tuple: x of shape [batch_size, height, width, channels] and y of shape [batch_size, *CELLS_SHAPE, 5].

        NOTE: Check for last batch is done by tensorflow Sequence superclass (no need to raise IndexError).
        """

        # Get file paths and box annotations for this batch.
        batch_info = self._parser.info[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        # Calculate the batch size (last batch may be smaller)
        this_batch_size = self.batch_size
        if batch_index == self.__len__() - 1:
            this_batch_size = len(self._parser.info) % self.batch_size

        # Initialize input and output arrays.
        x = np.empty(shape=(this_batch_size, *Dataset.IMAGE_SHAPE, 3), dtype=np.float32)
        y = np.empty(shape=(this_batch_size, *Dataset.CELLS_SHAPE, 5), dtype=np.float32)

        # Fill input and output arrays image-by-image.
        for index, (filename, box_list) in enumerate(batch_info):

            # Load the image and instantiate a numpy array of boxes (one per row).
            this_image = cv2.imread(self._parser.get_path(filename))
            boxes = np.array(box_list, dtype=np.float32)

            # Resize image to model input shape and rescale boxes accordingly.
            this_image, boxes = self._resize(this_image, boxes)

            # Normalize input image to 0-1.
            this_x = this_image / 255

            # Encode boxes into an output array.
            this_y = self._encoder.encode(boxes)

            # Append this x and y to the batch.
            x[index, ...] = this_x
            y[index, ...] = this_y

        # Limit y for type 0 (detected box in cell y/n) and 1 (+ 0-1 x, y shift from cell's left-upper corner) outputs.
        if self.type == 0:
            y = y[:, :, :, :1]
        if self.type == 1:
            y = y[:, :, :, :3]

        # Return batch.
        # TODO make yield_y = False mode faster by not retrieving boxes.
        if self.yield_y:
            return x, y
        else:
            return x

    def on_epoch_end(self):
        """Shuffle dataset on epoch end if shuffle flag is True."""
        if self.shuffle:
            # TODO create a self.indices proxy so that shuffling is not done directly in _parse.info.
            random.shuffle(self._parser.info)

    @staticmethod
    def _resize(image, boxes):
        """
        Resize input image and associated bounding boxes to fit the model input shape.

        :param np.ndarray image: image to resize.
        :param np.ndarray boxes: bounding boxes in the row format [x1, y1, w, h] (upper left corner, width and height).
        :return tuple: images resized to class resize shape (model input shape) and bounding boxes resized accordingly.
        """
        height, width, _ = image.shape
        boxes = BoxEncoder.scale(boxes=boxes, from_shape=(height, width), to_shape=Dataset.IMAGE_SHAPE)
        image = cv2.resize(image, dsize=tuple(reversed(Dataset.IMAGE_SHAPE)))  # Use opencv convention (width, height).
        return image, boxes

    def show(self, resize=False):
        """
        Look at the images and corresponding bounding boxes one by one (press 'q' to exit).

        :param bool resize: if True resize images and bounding boxes to IMAGE_SHAPE (otherwise use original shape).
        """

        # Creates generator that loads each image in sequence and instantiate a numpy array for its boxes.
        generator = (
            (cv2.imread(self._parser.get_path(filename)), np.array(box_list, dtype=np.float32))
            for filename, box_list in self._parser.info
        )

        # Resize image and boxes to model input shape.
        if resize:
            generator = (self._resize(image, boxes) for image, boxes in generator)

        # Show the image and boxes.
        Plotter.show(generator)


if __name__ == "__main__":
    """Run a few quick tests for the dataset generator."""
    ds = Dataset(dataset="validation")
    ds.test(batch_index=1)
    ds.show(resize=False)  # Press 'q' to quit displaying images.
    print(f"Number of batches in validation dataset: {len(ds)}")
    for batch_number, (x, y) in enumerate(ds):
        print(f"batch {batch_number}: x.shape={x.shape}, y.shape={y.shape}")
