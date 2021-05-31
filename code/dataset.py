import random
import math
import numpy as np
import cv2
import tensorflow as tf
from coco_anno_reader_fl import get_input_for_fl


class Parser:
    """Parse dataset annotations and navigate data folder."""

    # Dataset parent directory.
    PARENT_DIR = "../data"

    # Train and validation sets subdirectories.
    DATA_DIR_TRAIN = "train2017"
    DATA_DIR_VALIDATION = "val2017"

    # Annotations directory and filename for each set.
    ANNOTATIONS_DIR = "annotations"
    ANNOTATIONS_FILENAME_TRAIN = "instances_train2017.json"
    ANNOTATIONS_FILENAME_VALIDATION = "instances_val2017.json"

    def __init__(self, dataset):
        """
        :param str dataset: either "train" or "validation".
        """
        assert dataset == "train" or dataset == "validation", "Dataset input can be either 'train' or 'validation'."
        self._info = Parser._parse_annotations(dataset=dataset)
        self._data_dir = Parser._parse_data_directory(dataset=dataset)

    def get_path(self, filename):
        """
        :param str filename: name of an image file in the data folder.
        :return str: full path to the image file (to be used as input to image show method) assuming file exists.
        """
        return f"{self._data_dir}/{filename}"

    @property
    def info(self):
        """:return list: lists with filenames and their associated list of ground truth detection bounding boxes."""
        return self._info

    @staticmethod
    def _parse_annotations(dataset):
        if dataset == "train":
            info = get_input_for_fl(
                json_path=f"{Parser.PARENT_DIR}/{Parser.ANNOTATIONS_DIR}/{Parser.ANNOTATIONS_FILENAME_TRAIN}"
            )
        else:
            info = get_input_for_fl(
                json_path=f"{Parser.PARENT_DIR}/{Parser.ANNOTATIONS_DIR}/{Parser.ANNOTATIONS_FILENAME_VALIDATION}"
            )
        return info

    @staticmethod
    def _parse_data_directory(dataset):
        if dataset == "train":
            data_dir = f"{Parser.PARENT_DIR}/{Parser.DATA_DIR_TRAIN}"
        else:
            data_dir = f"{Parser.PARENT_DIR}/{Parser.DATA_DIR_VALIDATION}"
        return data_dir


class Dataset(tf.keras.utils.Sequence):
    """Generator of image batches and their ground truth outputs for training a one-step object detection model."""

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
            this_y = self._encode(boxes)

            # Append this x and y to the batch.
            x[index, ...] = this_x
            y[index, ...] = this_y

        # Limit y for type 0 (detected box in cell y/n) and 1 (+ 0-1 x, y shift from cell's left-upper corner) outputs.
        if self.type == 0:
            y = y[:, :, :, 0]
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

        # Extract old and new image width and height (note inverted shape conventions between numpy and opencv).
        this_height, this_width, _ = image.shape
        height, width = Dataset.IMAGE_SHAPE

        # Resize bounding boxes [x1, y1, w, h].
        boxes[:, [0, 2]] *= width / this_width
        boxes[:, [1, 3]] *= height / this_height

        # Resize image.
        image = cv2.resize(image, dsize=tuple(reversed(Dataset.IMAGE_SHAPE)))  # Use opencv convention (width, height).

        # Return resized image and its corresponding bounding boxes.
        return image, boxes

    @staticmethod
    def _encode(boxes):
        """
        Encode array of bounding boxes into its corresponding output array in cell-normalized units. Each bounding
        box is encoded into the output array as a 1 in the first channel of the "responsible" cell followed by 4
        channels that contain the box parameters in the range 0-1: the box upper left corner's shift relative to the
        cell upper left corner, and its width and height relative to those of the input image.

        :param np.ndarray boxes: (batch_size, 4) array where each row represents a box in the format [x1, y1, w, h].
        :return np.ndarray: box-encoded output array in the shape (batch_size, *CELLS_SHAPE, 5).
        """

        # Extract input and output width and height.
        input_height, input_width = Dataset.IMAGE_SHAPE
        output_height, output_width = Dataset.CELLS_SHAPE

        # Rescale upper left box corner coordinates x1 and y1 relative to output (cells) shape.
        boxes[:, 0] *= output_width / input_width
        boxes[:, 1] *= output_height / input_height

        # Rescale box width and height relative to input (image) shape (values in 0-1).
        boxes[:, 2] /= input_width
        boxes[:, 3] /= input_height

        # Extract "responsible" cell indices [cell_col_index, cell_row_index].
        cell_indices = np.floor(boxes[:, :2]).astype(int)

        # Shift box corner coordinates to count from their responsible cell's upper left corner (values in 0-1).
        boxes[:, :2] = np.mod(boxes[:, :2], 1)

        # Encode boxes in an output array.
        # TODO vectorize this.
        this_y = np.zeros(shape=(*Dataset.CELLS_SHAPE, 5))
        for (col, row), box in zip(cell_indices, boxes):
            this_y[row, col, 0] = 1  # 1 means a box is associated with this cell.
            this_y[row, col, 1:] = box  # Upper left corner x1, y1 in 0-1 measured from cell corner & width and height.

        # Return output array.
        return this_y

    @staticmethod
    def _decode(this_y):
        """
        Decode a single sample output y into bounding boxes.

        :param np.ndarray this_y: (*CELLS_SHAPE, 5) array [detected y/n, cell-normalized x1, y1, image-normalized w, h].
        """

        # Get the row and column indices of cells associated with boxes (non-zero score).
        rows, cols = np.nonzero(this_y[:, :, 0])

        # Instantiate the array of boxes.
        boxes = np.empty(shape=(rows.size, 4), dtype=np.float32)

        # Extract input and output width and height.
        input_height, input_width = Dataset.IMAGE_SHAPE
        output_height, output_width = Dataset.CELLS_SHAPE

        # TODO vectorize this.
        # Fill box parameters one-by-one in the form [x1, y1, w, h] measured in input image pixels.
        for index, (row, col) in enumerate(zip(rows, cols)):
            x1 = (col + this_y[row, col, 1]) * input_width / output_width
            y1 = (row + this_y[row, col, 2]) * input_height / output_height
            w = this_y[row, col, 3] * input_width
            h = this_y[row, col, 4] * input_height
            boxes[index, ...] = np.array([x1, y1, w, h])

        # Return boxes array.
        return boxes

    @staticmethod
    def _draw_boxes(image, boxes):
        """
        Draw boxes one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw boxes on (assumed uint8 BGR image).
        :param np.ndarray boxes: boxes array sized [num_boxes, x1, y1, width, height] in image pixel units.
        """

        # Create a random color scheme.
        colors = 255 * np.random.rand(32, 3)

        # Draw boxes one-by-one.
        for index, box in enumerate(boxes):
            # Unpack box and extract corners (1: upper left, 2: lower right).
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h

            # Draw a bounding box onto the image.
            image = cv2.rectangle(
                image,
                (int(x1), int(y1)),  # Upper left corner.
                (int(x2), int(y2)),  # Lower right corner.
                color=colors[index % 32, :],
                thickness=2
            )

    def test(self, batch_index=0):
        """
        Test the encoding/decoding of annotation boxes to y output for a given batch index.

        :param int batch_index: index of the batch to test (all) bounding boxes for.
        """

        # Get batch corresponding to input index using __getitem__.
        assert self.type == 2, "No testing on partial output y."
        _, y = self[batch_index]

        # Get the batch annotation and path data.
        batch_info = self._parser.info[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        # Test equality of annotation and encoded/decoded boxes for each image.
        for this_y, (filename, box_list) in zip(y, batch_info):

            # Extract image and detection bounding boxes.
            boxes_out = self._decode(this_y)

            # Load the image and boxes and resize to model input shape.
            _, boxes_in = self._resize(
                image=cv2.imread(self._parser.get_path(filename)),
                boxes=np.array(box_list, dtype=np.float32)
            )

            # Test output/input box equality for all input boxes other than those sharing the same cell (one per cell).
            # TODO Make sure all input boxes that are omitted share same cell with ones not omitted.
            if boxes_in.shape != boxes_out.shape:
                for box in boxes_out:
                    # Check if box contains in annotation boxes.
                    assert np.isclose(boxes_in, box).all(axis=1).any()
            else:
                np.testing.assert_array_almost_equal(np.sort(boxes_in, axis=0), np.sort(boxes_out, axis=0), decimal=4)

    def show(self, resize=False):
        """
        Look at the images and corresponding bounding boxes one by one (press 'q' to exit).

        :param bool resize: if True resize images and bounding boxes to IMAGE_SHAPE (otherwise use original shape).
        """

        # Show each image along with its bounding boxes in sequence.
        window_name = "decoded-y vs. annotation boxes"
        for filename, box_list in self._parser.info:

            # Load the image and instantiate a numpy array of boxes (one per row)..
            image = cv2.imread(self._parser.get_path(filename))
            boxes = np.array(box_list, dtype=np.float32)

            # Resize image and boxes to model input shape.
            if resize:
                image, boxes = self._resize(image, boxes)

            # Draw bounding boxes on the image.
            self._draw_boxes(image=image, boxes=boxes)

            # Show the image with drawn bounding boxes.
            cv2.imshow(window_name, image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    """Run a few quick tests for the dataset generator."""
    ds = Dataset(dataset="validation")
    ds.test(batch_index=1)
    ds.show(resize=False)  # Press 'q' to quit displaying images.
    print(f"Number of batches in validation dataset: {len(ds)}")
    for batch_number, (x, y) in enumerate(ds):
        print(f"batch {batch_number}: x.shape={x.shape}, y.shape={y.shape}")
