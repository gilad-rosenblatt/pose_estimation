import numpy as np


# TODO implement IOU.
# TODO implement NMS.


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

        # Extract "responsible" cell indices [cell_col_index, cell_row_index].
        cell_indices = np.floor(boxes[:, :2]).astype(int)

        # Shift box center coordinates to count from their responsible cell's upper left corner (values in 0-1).
        boxes[:, :2] = np.mod(boxes[:, :2], 1)

        # Encode boxes in an output array.
        # TODO vectorize this.
        this_y = np.zeros(shape=(*self.cells_shape, 5))
        for (col, row), box in zip(cell_indices, boxes):
            this_y[row, col, 0] = 1  # 1 means a box is associated with this cell.
            this_y[row, col, 1:] = box  # Box center xc, yc in 0-1 measured from cell up-left corner + width & height.

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

        # Instantiate the array of boxes and scores.
        boxes = np.empty(shape=(rows.size, 4), dtype=np.float32)
        scores = np.empty(shape=(rows.size, 1), dtype=np.float32)

        # Extract input and output width and height.
        input_height, input_width = self.image_shape
        output_height, output_width = self.cells_shape

        # TODO vectorize this.
        # Fill box parameters one-by-one in the form [x1, y1, w, h] measured in input image pixels.
        for index, (row, col) in enumerate(zip(rows, cols)):
            xc = (col + this_y[row, col, 1]) * input_width / output_width  # Box center xc.
            yc = (row + this_y[row, col, 2]) * input_height / output_height  # Box center yc.
            w = this_y[row, col, 3] * input_width
            h = this_y[row, col, 4] * input_height
            boxes[index, ...] = np.array([xc - w / 2, yc - h / 2, w, h])  # Upper left corner: x,y1 = x,yc - w,h / 2.
            scores[index, ...] = this_y[row, col, 0]

        # Return boxes array.
        return boxes, scores

    def decode_centers(self, this_y):
        """
        Decode a single sample y prediction output in grid cell into "responsible" cell centers in input image pixels.

        :param np.ndarray this_y: (*CELLS_SHAPE, 5) array [detected y/n, cell-relative xc, yc, image-normalized w, h].
        :return np.ndarray: (num_boxes, 2) array where each row represents responsible cell centers in [xc, yc] format.
        """
        centered_y = np.zeros(shape=(*self.cells_shape, 5), dtype=np.float32)
        centered_y[..., 0] = this_y[..., 0]
        rows, cols = np.nonzero(this_y[:, :, 0])
        if this_y.shape[-1] < 3:
            centered_y[rows, cols, 1:3] = 0.5  # Missing data: set box upper left corner to cell's center (w, h = 0).
        else:
            centered_y[rows, cols, 1:3] = this_y[rows, cols, 1:3]  # Add shift relative to upper left corner of cell.
        boxes, _ = self.decode(this_y=centered_y)  # Return x1, y1 coordinates of decoded boxes (= xc, yc).
        return boxes[:, :2]

    def get_cells(self, centers):
        """
        Get the responsible cell indices for box centers given in image pixel coordinates.

        :param np.ndarray centers: a (num_points, 2) array of xc, yc box center points.
        :return np.ndarray: a (num_points, 2) array of "responsible" cell row, column indices.
        """

        # Extract old and new image width and height.
        from_height, from_width = self.image_shape
        to_height, to_width = self.cells_shape

        # Rescale bounding box centers [xc, yc] to gird cell units and floor to get responsible cell [row, col] indices.
        cells = np.empty(shape=centers.shape, dtype=int)
        cells[:, 1] = np.floor(centers[:, 0] * to_width / from_width).astype(int)  # # Column index.
        cells[:, 0] = np.floor(centers[:, 1] * to_height / from_height).astype(int)  # Row index.

        # Return cell indices.
        return cells

    @staticmethod
    def decode_scores(this_y):
        """
        Decode the box scores from single sample y prediction output in grid cell. The y input must have zeroes in
        the first channel where boxes are not predicted or 0-scored.

        :param np.ndarray this_y: (*CELLS_SHAPE, 5) array [detected y/n, cell-relative xc, yc, image-normalized w, h].
        :return np.ndarray: (num_boxes, 1) non-zero scores with rows ordered like the boxes output of the decode method.
        """

        # Get the row and column indices of cells associated with non-zero (predicted box) scores.
        rows, cols = np.nonzero(this_y[:, :, 0])

        # Instantiate the array of scores.
        scores = np.empty(shape=(rows.size, 1), dtype=np.float32)

        # TODO vectorize this.
        # Fill scores one-by-one.
        for index, (row, col) in enumerate(zip(rows, cols)):
            scores[index, ...] = this_y[row, col, 0]

        # Return array of box scores.
        return scores

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
