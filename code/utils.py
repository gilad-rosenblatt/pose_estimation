import numpy as np


class BoxEncoder:
    """Encoder for bounding boxes in an image divided into grid cells."""

    def __init__(self, image_shape, cells_shape):
        """
        Save input (image) and output (cells) grid dimensions.

        :param tuple image_shape: shape of input image.
        :param tuple cells_shape: shape of output grid cells.
        """
        self.image_shape = image_shape
        self.cells_shape = cells_shape

    def encode(self, boxes):
        """
        Encode array of bounding boxes into its corresponding output array in cell-normalized units. Each bounding
        box is encoded into the output array as a 1 in the first channel of the "responsible" cell followed by 4
        channels that contain the box parameters in the range 0-1: the box upper left corner's shift relative to the
        cell upper left corner, and its width and height relative to those of the input image.

        :param np.ndarray boxes: (batch_size, 4) array where each row represents a box in the format [x1, y1, w, h].
        :return np.ndarray: box-encoded output array in the shape (batch_size, *CELLS_SHAPE, 5).
        """

        # Extract input and output width and height.
        input_height, input_width = self.image_shape
        output_height, output_width = self.cells_shape

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
        this_y = np.zeros(shape=(*self.cells_shape, 5))
        for (col, row), box in zip(cell_indices, boxes):
            this_y[row, col, 0] = 1  # 1 means a box is associated with this cell.
            this_y[row, col, 1:] = box  # Upper left corner x1, y1 in 0-1 measured from cell corner & width and height.

        # Return output array.
        return this_y

    def decode(self, this_y):
        """
        Decode a single sample output y in output (cells) grid into bounding boxes in input (image) grid.

        :param np.ndarray this_y: (*CELLS_SHAPE, 5) array [detected y/n, cell-normalized x1, y1, image-normalized w, h].
        """

        # Get the row and column indices of cells associated with boxes (non-zero score).
        rows, cols = np.nonzero(this_y[:, :, 0])

        # Instantiate the array of boxes.
        boxes = np.empty(shape=(rows.size, 4), dtype=np.float32)

        # Extract input and output width and height.
        input_height, input_width = self.image_shape
        output_height, output_width = self.cells_shape

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
    def scale(boxes, from_shape, to_shape):
        """
        Scale bounding boxes to fit resized image.

        :param np.ndarray boxes: bounding boxes in the row format [x1, y1, w, h] (upper left corner, width and height).
        :param tuple from_shape: (height, width) of the image associated with input boxes.
        :param tuple to_shape: (height, width) of the image associates with scaled boxes.
        :return np.ndarray : bounding boxes in the row format [x1, y1, w, h] scaled to correspond to output image.
        """

        # Extract old and new image width and height.
        from_height, from_width = from_shape
        to_height, to_width = to_shape

        # Resize bounding boxes [x1, y1, w, h].
        boxes[:, [0, 2]] *= to_width / from_width
        boxes[:, [1, 3]] *= to_height / from_height

        # Return resized boxes.
        return boxes
