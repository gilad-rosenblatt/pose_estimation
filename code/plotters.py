import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

from boxops import NMS
from encoders import DetectionsEncoder
from parsers import DetectionsParser, KeypointsParser


class Drawer:
    """Drawer of common shapes on images."""

    # Colors to use for persistent color selection using _get_color.
    NUM_COLORS = 36
    COLORS = 255 * np.random.rand(NUM_COLORS, 3)

    @staticmethod
    def get_color(index):
        """
        Maps and index to a color (persistent mapping).

        :param int index: index to map to color.
        :return np.ndarray: BGR color in 0-255.
        """
        return Drawer.COLORS[index % Drawer.NUM_COLORS, :]

    @staticmethod
    def draw_boxes(image, boxes, colors=None, thickness=2):
        """
        Draw boxes one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray boxes: boxes array sized (num_boxes, 4) for x1, y1, width, height in image pixel units.
        :param list colors: list of BGR colors in 0-255 one for each input point.
        :param int thickness: line thickness to use.
        """
        if not colors:
            colors = [Drawer.get_color(index=index) for index, _ in enumerate(boxes)]
        for (x1, y1, w, h), color in zip(boxes, colors):
            image = cv2.rectangle(
                image,
                (int(x1), int(y1)),  # Upper left corner.
                (int(x1 + w), int(y1 + h)),  # Lower right corner.
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA
            )

    @staticmethod
    def draw_circles(image, points, colors=None, thickness=2, radius=12):
        """
        Draw circles one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray points: points array sized (num_points, 2) in image pixel units.
        :param list colors: list of BGR colors in 0-255 one for each input point.
        :param int thickness: line thickness to use.
        :param int radius: radius to use when drawing circles.
        """
        if not colors:
            colors = [Drawer.get_color(index=index) for index, _ in enumerate(points)]
        for (xc, yc), color in zip(points, colors):
            image = cv2.circle(
                image,
                center=(int(xc), int(yc)),  # Center point.
                radius=radius,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA
            )

    @staticmethod
    def draw_scores(image, scores, points, colors=None, thickness=1):
        """
        Put text containing scores one-by-one on the image (carries side effect to image array).

        :param np.ndarray scores: scores to write at input points of size (num_points,)
        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray points: scores array sized (num_points, 1).
        :param np.ndarray points: points array sized (num_points, 2) in image pixel units.
        :param list colors: list of BGR colors in 0-255 one for each input point.
        :param int thickness: line thickness to use.
        """
        if not colors:
            colors = [255 * np.array([1, 1, 1]) for _, _ in enumerate(points)]
        for score, (x_bl, y_bl), color in zip(scores, points, colors):
            image = cv2.putText(
                image,
                text=f"{float(score):.2f}",
                org=(int(x_bl), int(y_bl)),  # Bottom left corner of text string.
                color=color,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1 / 2,
                thickness=thickness
            )

    @staticmethod
    def draw_lines(image, lines, colors=None, thickness=2, radius=3, mark_endpoints=False):
        """
        Draw lines one-by-one on the image and mark their endpoints (carries side effect to image array).

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray lines: lines array sized (num_lines, 2, 2) in pixel units denoting start/end point (x, y).
        :param list colors: list of BGR colors in 0-255 one for each input line.
        :param int thickness: line thickness to use.
        :param int radius: radius to use when drawing endpoints.
        :param bool mark_endpoints: if True marks the endpoints of each line with circles.
        """
        if not colors:
            colors = [Drawer.get_color(index=index) for index, _ in enumerate(lines)]
        for (start_point, end_point), color in zip(lines, colors):
            image = cv2.line(
                image,
                start_point.astype(int),
                end_point.astype(int),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA
            )
            if mark_endpoints:
                points = np.vstack((start_point, end_point))
                Drawer.draw_circles(image=image, points=points, colors=[color] * 2, thickness=-1, radius=radius)


class Skeleton:
    """Representation of a person's keypoints skeleton."""

    # Interpretation of each of the 17 person keypoints by index.
    # FIXME missing 1 keypoint name.
    KEYPOINT_NAMES = [
        "nose"
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"
    ]

    # Skeleton lines encoded in keypoint indices (0-based) and their corresponding colors.
    SKELETON = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6]
    ]
    SKELETON_COLORS = [
        (int(b), int(g), int(r))
        for (r, g, b, a) in 255 * plt.cm.get_cmap(name="rainbow")(np.linspace(start=0, stop=1, num=len(SKELETON)))
    ]

    @staticmethod
    def _make_lines(keypoints):
        """
        Make skeleton lines out of an input keypoints array for a single person.

        :param np.ndarray keypoints: (num_keypoints=17, 3) array for x, y, visible_flag for each of a persons keypoints.
        :return np.ndarray: lines array sized (num_lines, 2, 2) in pixel units denoting start/end keypoints in (x, y).
        """
        lines = np.empty(shape=(len(Skeleton.SKELETON), 2, 2))
        colors = []
        keypoint_exists = keypoints[:, 2] != 0  # Visibility = 0 indicates a missing keypoint.
        index = 0
        for (start, end), color in zip(Skeleton.SKELETON, Skeleton.SKELETON_COLORS):
            if not keypoint_exists[start] or not keypoint_exists[end]:
                continue  # Do not use skeleton lines for missing keypoints.
            lines[index, 0, :] = keypoints[start, 0:2]
            lines[index, 1, :] = keypoints[end, 0:2]
            colors.append(color)
            index += 1
        return lines[:index], colors  # Return only valid indices.

    @staticmethod
    def draw(image, keypoints, box=None):
        """
        Make skeleton lines out of an input keypoints array for a single person over a given image. Optionally add box.

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray keypoints: (num_keypoints=17, 3) array for x, y, visible_flag for each of a persons keypoints.
        :param np.ndarray box: box array sized (4,) for x1, y1, width, height in image pixel units.
        """
        if isinstance(box, np.ndarray):
            yellow = (0, 204, 204)
            Drawer.draw_boxes(image=image, boxes=box[np.newaxis, :], colors=[yellow])
        lines, colors = Skeleton._make_lines(keypoints=keypoints)
        Drawer.draw_lines(image=image, lines=lines, colors=colors, mark_endpoints=True)


class DetectionsPlotter:
    """Plotter for images with bounding boxes."""

    @staticmethod
    def show_generator(generator):
        """
        Show the generated images and their bounding box annotations image by image.

        :param generator: generator of single image, boxes pairs.
        """

        # Show each image along with its bounding boxes in sequence.
        window_name = "bounding_boxes"
        for image, boxes in generator:

            # Draw bounding boxes on the image.
            Drawer.draw_boxes(image=image, boxes=boxes)

            # Show the image with drawn bounding boxes.
            cv2.imshow(window_name, image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)

    @staticmethod
    def show_batch(x, y_true, y_pred, nms_threshold=0.3):
        """
        Show the batch ground truth and predictions image by image.

        :param np.nd.array x: a (batch_size, *image_shape, 3) array of normalized (values in 0-1) images.
        :param np.nd.array y_true: (batch_size, *cells_shape, 5) ground-truth encoding bounding boxes and 0/1 scores.
        :param np.nd.array y_pred: (batch_size, *cells_shape, 5) model output encoding bounding boxes and scores.
        :param float nms_threshold: threshold used for non-max suppression when post processing the prediction boxes.
        """

        # Get input and output dimensions.
        _, *image_shape, _ = x.shape
        _, *cells_shape, _ = y_true.shape

        # Initialize encoder.
        encoder = DetectionsEncoder(input_shape=image_shape, output_shape=cells_shape)

        # Show ground-truth and predictions for each image in sequence.
        window_name = "annotation_boxes"
        for this_x, this_y, this_pred in zip(x, y_true, y_pred):

            # Cast image to uint8.
            image = (this_x * 255).astype(np.uint8)

            # Decode bounding boxes and extract cell center points for predictions (after NMS) and ground truth.
            boxes_true, _ = encoder.decode(this_y=this_y)
            centers_true = encoder.get_box_centers(boxes=boxes_true)
            boxes_pred, scores = encoder.decode(this_y=this_pred)
            boxes_pred, scores = NMS.perform(boxes=boxes_pred, scores=scores, threshold=nms_threshold)
            centers_pred = encoder.get_box_centers(boxes=boxes_pred)

            # Get colors for boxes based on the output grid cell their center points fall in.
            cells1 = encoder.get_cells(centers=centers_true)
            cells2 = encoder.get_cells(centers=centers_pred)
            colors1 = [Drawer.get_color(index=index) for index in np.ravel_multi_index(cells1.T, dims=cells_shape)]
            colors2 = [Drawer.get_color(index=index) for index in np.ravel_multi_index(cells2.T, dims=cells_shape)]

            # Draw bounding boxes and box center circles.
            Drawer.draw_boxes(image=image, boxes=boxes_true, colors=colors1, thickness=1)
            Drawer.draw_circles(image=image, points=centers_true, colors=colors1, thickness=1, radius=12)
            Drawer.draw_boxes(image=image, boxes=boxes_pred, colors=colors2, thickness=2)
            Drawer.draw_circles(image=image, points=centers_pred, colors=colors2, thickness=4, radius=4)
            Drawer.draw_scores(image=image, scores=scores, points=centers_pred, colors=colors2, thickness=1)

            # Show the image with drawn bounding boxes and circles.
            cv2.imshow(window_name, image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)

    @staticmethod
    def show_annotations(annotations):
        """
        Show the images corresponding to the input annotations along with their bounding box annotations image by image.

        :param list annotations: list of similarly formatted dictionaries each is an annotation in COCO format.
        """

        # Load the COCO annotations associated with json results file.
        coco_gt = COCO(annotation_file=DetectionsParser.get_annotation_file(dataset="validation"))
        path_to_data = os.path.join(DetectionsParser.PARENT_DIR, DetectionsParser.get_data_dir(dataset="validation"))

        # Show each annotation drawn on it corresponding image in sequence.
        window_name = "bounding_boxes"
        for annotation in annotations:

            # Load the image corresponding to this annotation.
            image_dict = coco_gt.loadImgs(ids=annotation["image_id"])[0]
            this_image = cv2.imread(os.path.join(path_to_data, image_dict['file_name']))

            # Get the bounding box from the annotation.
            box = np.array(annotation["bbox"])
            Drawer.draw_boxes(image=this_image, boxes=box[np.newaxis, :])

            # Show the image with drawn bounding box.
            cv2.imshow(window_name, this_image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)


class KeypointsPlotter:

    @staticmethod
    def show_generator(generator):
        """
        Show the generated images and their person keypoints (skeleton) and bounding box annotations image by image.

        :param generator: generator of single image, a (4,) bounding box, and (17,) keypoints trio.
        """

        # Show each image along with its bounding boxes in sequence.
        window_name = "keypoints"
        for image, box, keypoints in generator:

            # Draw skeleton with bounding box on the image.
            Skeleton.draw(image=image, keypoints=keypoints, box=box)

            # Show the image with drawn bounding boxes.
            cv2.imshow(window_name, image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)


if __name__ == "__main__":

    # Parse keypoints COCO annotations.
    parser = KeypointsParser(dataset="validation")

    # Creates generator that loads each image in sequence and instantiate a numpy array for its keypoints and boxes.
    generator = (
        (
            cv2.imread(filename=parser.get_path(filename=filename)),
            np.array(box, dtype=np.float32 ),
            np.array(keypoints, dtype=np.float32).reshape(-1, 3)
        )
        for filename, box, keypoints in parser.info
    )

    # Show the image and skeletons.
    KeypointsPlotter.show_generator(generator)
