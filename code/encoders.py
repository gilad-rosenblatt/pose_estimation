from abc import ABC, abstractmethod

import numpy as np
import cv2
import matplotlib.pyplot as plt

from parsers import KeypointsParser
from plotters import Skeleton, Drawer


class DataEncoder(ABC):
    """Encoder annotations data in an image for a convolutional neural network."""

    def __init__(self, input_shape, output_shape):
        """
        Save input (image) and network output (grid) dimensions.

        :param tuple input_shape: shape of network input image (height, width).
        :param tuple output_shape: shape of network output grid (height, width).
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encode image annotations data into network output representation (for producing ground truth at train)."""
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """Decode image annotations data from network output representation (for prediction/inference at test)."""
        pass


class DetectionsEncoder(DataEncoder):
    """Encoder for detection bounding boxes in an image for an object detection network."""

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
        output_height, output_width = self.output_shape

        # Scale boxes to output grid cell size.
        boxes = DetectionsEncoder.scale_boxes(boxes=boxes, from_shape=self.input_shape, to_shape=self.output_shape)

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
        this_y = np.zeros(shape=(*self.output_shape, 5))
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
        input_height, input_width = self.input_shape
        output_height, output_width = self.output_shape

        # Fill box parameters in the form [x1, y1, w, h] measured in input image pixels.
        xc = (cols[:, np.newaxis] + this_y[rows, cols, 1:2]) * input_width / output_width  # Box center xc.
        yc = (rows[:, np.newaxis] + this_y[rows, cols, 2:3]) * input_height / output_height  # Box center yc.
        w = this_y[rows, cols, 3:4] * input_width
        h = this_y[rows, cols, 4:5] * input_height
        boxes = np.concatenate((xc - w / 2, yc - h / 2, w, h), axis=-1)  # Upper left corner x1,y1 = xc,yc - w,h / 2.
        scores = this_y[rows, cols, 0:1].copy()  # Do not return a slice (prevent unwanted side effects).

        # Return boxes and scores.
        return boxes, scores

    def get_cells(self, centers):
        """
        Get the responsible cell indices for box centers given in image pixel coordinates.

        :param np.ndarray centers: a (num_points, 2) array of xc, yc box center points.
        :return np.ndarray: a (num_points, 2) array of "responsible" cell row, column indices.
        """

        # Extract old and new image width and height.
        from_height, from_width = self.input_shape
        to_height, to_width = self.output_shape

        # Rescale bounding box centers [xc, yc] to grid cell units and floor to get responsible cell [row, col] indices.
        cells = np.empty(shape=centers.shape, dtype=int)
        cells[:, 1] = np.floor(centers[:, 0] * to_width / from_width).astype(int)  # Columns.
        cells[:, 0] = np.floor(centers[:, 1] * to_height / from_height).astype(int)  # Rows.

        # Return cell indices.
        return cells

    @staticmethod
    def get_box_centers(boxes):
        """
        Get box center points from boxes.

        :param np.ndarray boxes: (num_boxes, 4) array where rows are boxes in [x1, y1, w, h] (1: upper-left corner).
        :return np.ndarray: a (num_points, 2) array of xc, yc box center points.
        """
        return np.concatenate([boxes[..., 0:1] + boxes[..., 2:3] / 2, boxes[..., 1:2] + boxes[..., 3:] / 2], axis=-1)

    @staticmethod
    def scale_boxes(boxes, from_shape, to_shape):
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

        # Rescale bounding boxes x1, y1, w, h.
        boxes_scaled = np.empty(shape=boxes.shape, dtype=np.float32)
        boxes_scaled[:, [0, 2]] = boxes[:, [0, 2]] * to_width / from_width
        boxes_scaled[:, [1, 3]] = boxes[:, [1, 3]] * to_height / from_height

        # Return scaled boxes.
        return boxes_scaled


class KeypointsEncoder(DataEncoder):
    """Encoder for person keypoints in an image for a pose estimation network."""

    def encode(self, *args, **kwargs):
        raise NotImplemented

    def decode(self, *args, **kwargs):
        raise NotImplemented

    @staticmethod
    def expand_box(box, keypoints, image_shape, aspect_ratio):
        """
        Expand the input bounding box (never reduced, unless protruding from the image) to fit all keypoints and the
        target aspect ratio (if possible). Boxes and keypoints are tested for viability and bad cases raise exceptions.

        :param np.ndarray box: (4,) bounding box for in x1, y1, w, h format (1: upper left corner).
        :param np.ndarray keypoints: (num_keypoints, 3) keypoints array in x, y, visible format (0s for missing points).
        :param tuple image_shape: (image_height, image_width) of the image to which all x, y coordinates refer to.
        :param aspect_ratio: target aspect ratio for the output bounding box (not guaranteed).
        :return: (4,) expanded bounding box in same coordinates as input box given as x1, y1, w, h.
        """

        # Extract original image shape.
        image_height, image_width = image_shape

        # Unpack the box and assert it is valid. TODO move assert to parser?
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h  # This must be done before grooming.
        assert h > 0 and w > 0 and x2 > 0 and y2 > 0, "Ill defined bounding box (negative h/w or flipped corners)."

        # Unpack keypoints bounding box (minimal box that includes all keypoints). TODO Move assert tp parser?
        viable_keypoints = keypoints[keypoints[:, 2] != 0]  # Visibility = 0 indicates a missing keypoint.
        x1_kp = np.min(viable_keypoints[:, 0])
        y1_kp = np.min(viable_keypoints[:, 1])
        x2_kp = np.max(viable_keypoints[:, 0])
        y2_kp = np.max(viable_keypoints[:, 1])
        assert x1_kp >= 0 and y1_kp >= 0 and x2_kp <= image_width - 1 and y2_kp <= image_height - 1, "Keypoint outside."

        # Expand box to include all keypoints if it does not already.
        x1 = min(x1, x1_kp)
        y1 = min(y1, y1_kp)
        x2 = max(x2, x2_kp)
        y2 = max(y2, y2_kp)

        # Groom box to fit inside the original image (cannot crop outside-image pixel values).
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image_width - 1)  # 0-based.
        y2 = min(y2, image_height - 1)  # 0-based.

        # Represent box in centroid format (so that expand w/h operations are symmetric to center by default).
        w = x2 - x1
        h = y2 - y1
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2

        # Enforce aspect ratio by expanding box in the dimension it falls short (prefer symmetric-to-center expansion).
        if w / h > aspect_ratio:
            # Height should be increased.
            new_height = w / aspect_ratio
            if new_height > image_height - 1:
                # TOO BIG: crop will distort at resize.
                y1 = 0
                y2 = image_height - 1
            elif y_c + new_height / 2 > image_height - 1:
                # Center shifts up.
                y1 = image_height - 1 - new_height
                y2 = image_height - 1
            elif y_c - new_height / 2 < 0:
                # Center shifts down.
                y1 = 0
                y2 = new_height
            h = min(new_height, image_height - 1)
            y_c = (y1 + y2) / 2  # If center unchanged this is still true.
        else:
            # Width should be increased.
            new_width = h * aspect_ratio
            if new_width > image_width - 1:
                # TOO BIG: crop will distort at resize.
                x1 = 0
                x2 = image_width - 1
            elif x_c + new_width / 2 > image_width - 1:
                # Center shifts left.
                x1 = image_width - 1 - new_width
                x2 = image_width - 1
            elif x_c - new_width / 2 < 0:
                # Center shifts right.
                x1 = 0
                x2 = new_width
            w = min(new_width, image_width - 1)
            x_c = (x1 + x2) / 2  # If center unchanged this is still true.

        # Return normalized box in (x1, x2, w, h) format.
        return np.array([x_c - w / 2, y_c - h / 2, w, h])

    @staticmethod
    def move_keypoints(box, keypoints):
        """
        Translate keypoints to coordinates of a cropped part of the image denoted by a bounding box.

        :param np.ndarray box: (4,) crop bounding box x1, y1, w, h (1: upper left corner) in original image coordinates.
        :param np.ndarray keypoints: (num_keypoints, 3) keypoints in original image coordinates in x, y, visible format.
        :return: (num_keypoints=17, 3) keypoints in un-resized crop coordinate system in x, y, visible format.
        """

        # Instantiate an array for keypoints in the crop coordinates (mark all as missing with 0s for visibility).
        keypoints_translated = np.zeros(shape=keypoints.shape, dtype=np.float32)

        # Extract indices of viable keypoints (missing keypoints are denotes by visibility 0).
        is_keypoint = keypoints[:, 2] != 0

        # Copy the viable keypoints as-is (x, y, v).
        keypoints_translated[is_keypoint] = keypoints[is_keypoint]

        # Translate viable keypoints x, y w.r.t. crop upper left corner x1, y1 (do not touch v).
        keypoints_translated[is_keypoint, :2] -= box[:2]

        # Return translated keypoints.
        return keypoints_translated

    @staticmethod
    def scale_keypoints(keypoints, from_shape, to_shape):
        """
        Map keypoints given in crop coordinates to those of the network input, including resize and possibly distortion.

        :param np.ndarray keypoints: (num_keypoints, 3) keypoints in crop coordinates in x, y, visible format.
        :param tuple from_shape: (height, width) of the un-resized crop (crop of original image pixels).
        :param tuple to_shape: (height, width) of the network input (resized crop).
        :return: (num_keypoints=17, 3) keypoints in network input (resized crop) coordinates in x, y, visible format.
        """

        # Extract old and new crop width and height.
        from_height, from_width = from_shape
        to_height, to_width = to_shape

        # Instantiate an array for keypoints in the crop coordinates (mark all as missing with 0s for visibility).
        keypoints_scaled = np.zeros(shape=keypoints.shape, dtype=np.float32)

        # Extract indices of viable keypoints (missing keypoints are denotes by visibility 0).
        is_keypoint = keypoints[:, 2] != 0

        # Copy the viable keypoints as-is (x, y, v).
        keypoints_scaled[is_keypoint] = keypoints[is_keypoint]

        # Rescale keypoints x, y (do not touch v).
        keypoints_scaled[is_keypoint, 0] *= to_width / from_width
        keypoints_scaled[is_keypoint, 1] *= to_height / from_height

        # Return scaled keypoints.
        return keypoints_scaled

    @staticmethod
    def create_heatmap(keypoints, shape):

        # Instantiate a heatmap to all zeros (for missing keypoints).
        heatmap = np.zeros(shape=(*shape, 17), dtype=np.float32)  # TODO define Skeleton.NUM_KEYPOINTS?

        # Extract indices of viable keypoints (missing keypoints are denotes by visibility 0).
        is_keypoint = keypoints[:, 2] != 0
        viable_keypoints = keypoints[is_keypoint]

        # Define a mesh grid over pixel coordinate.
        height, width = shape
        x, y = np.meshgrid(
            np.linspace(0, width, width, endpoint=True),
            np.linspace(0, height, height, endpoint=True)
        )

        # Build a keypoint-centered Gaussian heatmap over the pixel grid where each keypoint is given a channel.
        # sigma = height * np.array([  # TODO define Skeleton.SIGMAS? Use them here?
        #     .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
        # ])[is_keypoint][np.newaxis, np.newaxis, :] / 10
        sigma = 3
        x_kps = viable_keypoints[:, 0]
        y_kps = viable_keypoints[:, 1]
        heatmap[..., is_keypoint] = np.exp(
            - ((((x[..., np.newaxis] - x_kps[np.newaxis, np.newaxis, :]) / sigma) ** 2) / 2)
            - ((((y[..., np.newaxis] - y_kps[np.newaxis, np.newaxis, :]) / sigma) ** 2) / 2)
        )

        # Return 1-per-keypoint heatmap.
        return heatmap


def main():

    # Instantiate a parser and encoder.
    parser = KeypointsParser(dataset="validation")
    encoder = KeypointsEncoder(input_shape=(256, 192), output_shape=(128, 96))

    # Create data generator.
    generator = (
        (cv2.imread(filename=parser.get_path(filename=filename)), np.array(box), np.array(keypoints).reshape(-1, 3))
        for filename, box, keypoints in parser.info
    )

    # Show each image along with its bounding boxes in sequence.
    window_names = ["annotation", "input", "output"]
    for image, box_detection, keypoints in generator:

        # Expand the detection box to get the crop box.
        input_height, input_width = encoder.input_shape
        aspect_ratio = input_width / input_height
        box_crop = encoder.expand_box(box=box_detection, keypoints=keypoints, image_shape=image.shape[:2], aspect_ratio=aspect_ratio)
        x1, y1, w, h = box_crop
        print(f"AR={w / h:.2f}, [{x1:.1f}, {y1:.1f}, {w:.1f}, {h:.1f}], {image.shape[:2]}")

        # Move keypoints and cut the crop.
        keypoints_moved = encoder.move_keypoints(box=box_crop, keypoints=keypoints)
        crop = image[int(y1):int(y1 + h), int(x1):int(x1 + w), ...]
        crop_shape = crop.shape[:2]

        # Scale keypoints and resize the crop to input shape.
        keypoints_scaled = encoder.scale_keypoints(keypoints=keypoints_moved, from_shape=crop_shape, to_shape=encoder.input_shape)
        crop_scaled = cv2.resize(crop, dsize=tuple(reversed(encoder.input_shape)))  # Opencv convention (w, h).

        # Scale keypoints and resize the crop to output shape.
        keypoints_output = encoder.scale_keypoints(keypoints=keypoints_moved, from_shape=crop_shape, to_shape=encoder.output_shape)
        crop_output = cv2.resize(crop, dsize=tuple(reversed(encoder.output_shape)))  # Opencv convention (w, h).

        # Draw skeletons with bounding boxes on the image and crop.
        Skeleton.draw(image=crop_scaled, keypoints=keypoints_scaled)
        Skeleton.draw(image=crop_output, keypoints=keypoints_output)
        Skeleton.draw(image=image, keypoints=keypoints, box=box_detection)
        Drawer.draw_boxes(image=image, boxes=box_crop[np.newaxis, :], colors=[(255, 255, 255)])

        # Show the image with drawn bounding boxes.
        cv2.imshow(window_names[0], image)
        cv2.imshow(window_names[1], crop_scaled)
        cv2.imshow(window_names[2], crop_output)

        # Break the loop if key 'q' was pressed.
        if cv2.waitKey() & 0xFF == ord("q"):
            break

        # Calculate the output crop heatmap.
        heatmap = encoder.create_heatmap(keypoints=keypoints_output, shape=encoder.output_shape)
        plt.figure()
        plt.imshow(cv2.cvtColor(crop_output, cv2.COLOR_BGRA2RGB))
        plt.imshow(heatmap.sum(axis=-1), alpha=0.7, interpolation="bilinear", cmap=plt.cm.get_cmap("viridis"))
        plt.show()

    # Close the windows.
    for window_name in window_names:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
