import os
import cv2
import numpy as np
from pycocotools.coco import COCO

from boxops import NMS
from encoders import BoxEncoder
from parsers import DetectionParser


class Plotter:
    """Plotter for images with bounding boxes."""

    # Colors to use for persistent color selection using _get_color.
    NUM_COLORS = 36
    COLORS = 255 * np.random.rand(NUM_COLORS, 3)

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
            Plotter._draw_boxes(image=image, boxes=boxes)

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
        encoder = BoxEncoder(image_shape=image_shape, cells_shape=cells_shape)

        # Show ground-truth and predictions for each image in sequence.
        window_name = "annotation_boxes"
        for this_x, this_y, this_pred in zip(x, y_true, y_pred):

            # Cast image to uint8.
            image = (this_x * 255).astype(np.uint8)

            # Decode bounding boxes and extract cell center points for predictions (after NMS) and ground truth.
            boxes_true, _ = encoder.decode(this_y=this_y)
            centers_true = encoder.get_centers(boxes=boxes_true)
            boxes_pred, scores = encoder.decode(this_y=this_pred)
            boxes_pred, scores = NMS.perform(boxes=boxes_pred, scores=scores, threshold=nms_threshold)
            centers_pred = encoder.get_centers(boxes=boxes_pred)

            # Get colors for boxes based on the output grid cell their center points fall in.
            cells1 = encoder.get_cells(centers=centers_true)
            cells2 = encoder.get_cells(centers=centers_pred)
            colors1 = [Plotter._get_color(index=index) for index in np.ravel_multi_index(cells1.T, dims=cells_shape)]
            colors2 = [Plotter._get_color(index=index) for index in np.ravel_multi_index(cells2.T, dims=cells_shape)]

            # Draw bounding boxes and box center circles.
            Plotter._draw_boxes(image=image, boxes=boxes_true, colors=colors1, thickness=1)
            Plotter._draw_circles(image=image, points=centers_true, colors=colors1, thickness=1, radius=12)
            Plotter._draw_boxes(image=image, boxes=boxes_pred, colors=colors2, thickness=2)
            Plotter._draw_circles(image=image, points=centers_pred, colors=colors2, thickness=4, radius=4)
            Plotter._draw_scores(image=image, scores=scores, points=centers_pred, colors=colors2, thickness=1)

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
        coco_gt = COCO(annotation_file=DetectionParser.get_annotation_file(dataset="validation"))
        path_to_data = os.path.join(DetectionParser.PARENT_DIR, DetectionParser.get_data_dir(dataset="validation"))

        # Show each annotation drawn on it corresponding image in sequence.
        window_name = "bounding_boxes"
        for annotation in annotations:

            # Load the image corresponding to this annotation.
            image_dict = coco_gt.loadImgs(ids=annotation["image_id"])[0]
            this_image = cv2.imread(os.path.join(path_to_data, image_dict['file_name']))

            # Get the bounding box from the annotation.
            box = np.array(annotation["bbox"])
            Plotter._draw_boxes(image=this_image, boxes=box[np.newaxis, :])

            # Show the image with drawn bounding box.
            cv2.imshow(window_name, this_image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)

    @staticmethod
    def _draw_boxes(image, boxes, colors=None, thickness=2):
        """
        Draw boxes one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray boxes: boxes array sized [num_boxes, x1, y1, width, height] in image pixel units.
        :param list colors: list of BGR colors in 0-255 one for each input point.
        :param int thickness: line thickness to use.
        """
        if not colors:
            colors = [Plotter._get_color(index=index) for index, _ in enumerate(boxes)]
        for (x1, y1, w, h), color in zip(boxes, colors):
            image = cv2.rectangle(
                image,
                (int(x1), int(y1)),  # Upper left corner.
                (int(x1 + w), int(y1 + h)),  # Lower right corner.
                color=color,
                thickness=thickness
            )

    @staticmethod
    def _draw_circles(image, points, colors=None, thickness=2, radius=12):
        """
        Draw circles one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray points: points array sized [num_points, xc, yc] in image pixel units.
        :param list colors: list of BGR colors in 0-255 one for each input point.
        :param int thickness: line thickness to use.
        :param int radius: radius to use when drawing circles.
        """
        if not colors:
            colors = [Plotter._get_color(index=index) for index, _ in enumerate(points)]
        for (xc, yc), color in zip(points, colors):
            image = cv2.circle(
                image,
                center=(int(xc), int(yc)),  # Center point.
                radius=radius,
                color=color,
                thickness=thickness
            )

    @staticmethod
    def _draw_scores(image, scores, points, colors=None, thickness=1):
        """
        Put text containing scores one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw on (assumed uint8 BGR image).
        :param np.ndarray points: scores array sized [num_points, score].
        :param np.ndarray points: points array sized [num_points, xc, yc] in image pixel units.
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
                fontScale=1/2,
                thickness=thickness
            )

    @staticmethod
    def _get_color(index):
        """
        Maps and index to a color (persistent mapping).

        :param int index: index to map to color.
        :return np.ndarray: BGR color in 0-255.
        """
        return Plotter.COLORS[index % Plotter.NUM_COLORS, :]
