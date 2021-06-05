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
        box is encoded into the output array as 1 in the first channel of the "responsible" cell followed by 4
        channels that contain the box parameters in the range 0-1: the box center's shift relative to the
        responsible cell's upper left corner, and its normalized width and height relative to image size (0-1).

        :param np.ndarray boxes: (num_boxes, 4) array where rows are boxes in [x1, y1, w, h] (1: upper-left corner).
        :return np.ndarray: box-encoded output array in the shape (batch_size, *CELLS_SHAPE, 5).
        """

        # Extract grid cell width and height.
        output_height, output_width = self.cells_shape

        # Scale boxes to output grid cell size.
        boxes = BoxEncoder.scale(boxes=boxes, from_shape=self.image_shape, to_shape=self.cells_shape)

        # Convert upper left corner to box centers and normalize width and height.
        boxes[:, 0] += boxes[:, 2] / 2  # x1 --> xc = x1 + w / 2.
        boxes[:, 1] += boxes[:, 3] / 2  # y1 --> yc = y1 + h / 2.
        boxes[:, 2] /= output_width  # w in 0-1
        boxes[:, 3] /= output_height  # h in 0-1

        # Extract "responsible" cell indices [cell_column_index, cell_row_index].
        indices = np.floor(boxes[:, :2]).astype(int)
        rows, cols = indices[:, 1], indices[:, 0]

        # Shift box center coordinates to count from their responsible cell's upper left corner (values in 0-1).
        boxes[:, :2] = np.mod(boxes[:, :2], 1)

        # Encode boxes: 1 in 1st channel means a box [xc, yc, w, h] is associated with this cell.
        this_y = np.zeros(shape=(*self.cells_shape, 5))
        this_y[rows, cols] = np.concatenate((np.ones(shape=(boxes.shape[0], 1)), boxes), axis=1)

        # Return output array.
        return this_y

    def decode(self, this_y):
        """
        Decode a single sample y prediction output in grid cell into bounding boxes in input image pixels. The y
        input must have zeroes in the first channel where boxes are not predicted but can have non-1 values where
        they are (probabilities or confidence scores). Any non-zero value will be treated as a valid box prediction.

        :param np.ndarray this_y: (*CELLS_SHAPE, 5) array [detected y/n, cell-relative xc, yc, image-normalized w, h].
        :return tuple: (num_boxes, 4) boxes numpy array in [x1, y1, w, h] format and their (num_boxes, 1) scores array.
        """

        # Get the row and column indices of cells associated with boxes (non-zero score).
        rows, cols = np.nonzero(this_y[:, :, 0])

        # Extract input and output width and height.
        input_height, input_width = self.image_shape
        output_height, output_width = self.cells_shape

        # Fill box parameters in the form [x1, y1, w, h] measured in input image pixels.
        xc = (cols[:, np.newaxis] + this_y[rows, cols, 1:2]) * input_width / output_width  # Box center xc.
        yc = (rows[:, np.newaxis] + this_y[rows, cols, 2:3]) * input_height / output_height  # Box center yc.
        w = this_y[rows, cols, 3:4] * input_width
        h = this_y[rows, cols, 4:5] * input_height
        boxes = np.concatenate((xc - w / 2, yc - h / 2, w, h), axis=-1)  # Upper left corner x1,y1 = xc,yc - w,h / 2.
        scores = this_y[rows, cols, 0:1].copy()  # Do not return a slice (prevent unwanted side effects).

        # Return boxes and scores.
        return boxes, scores

    def decode_centers(self, partial_y):
        """
        Decode a partial single sample y prediction in grid cell into "responsible" cell centers in input image pixels.

        :param np.ndarray partial_y: (*CELLS_SHAPE, <=3) array [detected y/n, cell-relative xc, yc, normalized w, h].
        :return np.ndarray: (num_boxes, 2) array where each row represents responsible cell centers in [xc, yc] format.
        """

        # Instantiate a full single-sample y array.
        this_y = np.zeros(shape=(*self.cells_shape, 5), dtype=np.float32)

        # Copy the scores from the partial y array.
        this_y[..., 0] = partial_y[..., 0]

        # Get the row and column indices of cells associated with boxes (non-zero score).
        rows, cols = np.nonzero(partial_y[:, :, 0])

        # Fill box centers or use cell centers if no box centers are provided.
        if partial_y.shape[-1] < 3:
            this_y[rows, cols, 1:3] = 0.5  # Set box upper left corner to cell's center (w, h = 0).
        else:
            this_y[rows, cols, 1:3] = partial_y[rows, cols, 1:3]  # Add shift relative to cell's upper left corner.

        # Decode the full y array and return box centers and scores (x1, y1 coordinates of decoded boxes = xc, yc).
        boxes, scores = self.decode(this_y=this_y)
        return boxes[:, :2], scores

    def get_cells(self, centers):
        """
        Get the responsible cell indices for box centers given in image pixel coordinates.

        :param np.ndarray centers: a (num_points, 2) array of xc, yc box center points.
        :return np.ndarray: a (num_points, 2) array of "responsible" cell row, column indices.
        """

        # Extract old and new image width and height.
        from_height, from_width = self.image_shape
        to_height, to_width = self.cells_shape

        # Rescale bounding box centers [xc, yc] to grid cell units and floor to get responsible cell [row, col] indices.
        cells = np.empty(shape=centers.shape, dtype=int)
        cells[:, 1] = np.floor(centers[:, 0] * to_width / from_width).astype(int)  # Columns.
        cells[:, 0] = np.floor(centers[:, 1] * to_height / from_height).astype(int)  # Rows.

        # Return cell indices.
        return cells

    @staticmethod
    def get_centers(boxes):
        """
        Get box center points from boxes.

        :param np.ndarray boxes: (num_boxes, 4) array where rows are boxes in [x1, y1, w, h] (1: upper-left corner).
        :return np.ndarray: a (num_points, 2) array of xc, yc box center points.
        """
        return np.concatenate([boxes[..., 0:1] + boxes[..., 2:3] / 2, boxes[..., 1:2] + boxes[..., 3:] / 2], axis=-1)

    @staticmethod
    def scale(boxes, from_shape, to_shape):
        """
        Scale bounding boxes to fit resized image.

        :param np.ndarray boxes: bounding boxes in the row format [x1, y1, w, h] (upper left corner, width and height).
        :param tuple from_shape: (height, width) of the image associated with input boxes.
        :param tuple to_shape: (height, width) of the image associates with scaled boxes.
        :return np.ndarray: bounding boxes in the row format [x1, y1, w, h] scaled to correspond to output image.
        """

        # Extract old and new image width and height.
        from_height, from_width = from_shape
        to_height, to_width = to_shape

        # Rescale bounding boxes [x1, y1, w, h].
        boxes_scaled = np.empty(shape=boxes.shape, dtype=np.float32)
        boxes_scaled[:, [0, 2]] = boxes[:, [0, 2]] * to_width / from_width
        boxes_scaled[:, [1, 3]] = boxes[:, [1, 3]] * to_height / from_height

        # Return scaled boxes.
        return boxes_scaled
