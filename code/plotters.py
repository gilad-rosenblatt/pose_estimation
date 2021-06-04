import cv2
import numpy as np
from encoder import BoxEncoder
from boxops import NMS


class Plotter:
    """Plotter for images with bounding boxes."""

    # Colors to use for persistent color selection using _get_color.
    NUM_COLORS = 36
    COLORS = 255 * np.random.rand(NUM_COLORS, 3)

    @staticmethod
    def show(generator):

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
    def show_batch(x, y_true, y_pred):

        _, *image_shape, _ = x.shape
        _, *cells_shape, _ = y_true.shape
        *_, num_output_channels = y_pred.shape
        encoder = BoxEncoder(image_shape=image_shape, cells_shape=cells_shape)

        # Show each image along with its bounding boxes in sequence.
        window_name = "annotation_boxes"
        for this_x, this_y, this_pred in zip(x, y_true, y_pred):

            # Cast image to uint8.
            image = (this_x * 255).astype(np.uint8)

            # Decode ground-truth bounding boxes and extract cell center points for predictions and ground truth.
            boxes_gt, _ = encoder.decode(this_y=this_y)
            centers_gt = encoder.get_centers(boxes=boxes_gt)
            if num_output_channels == 5:
                boxes, scores = encoder.decode(this_y=this_pred)
                boxes, scores = NMS.perform(boxes=boxes, scores=scores, threshold=0.3)
                centers = encoder.get_centers(boxes=boxes)
            else:
                centers = encoder.decode_centers(this_y=this_pred)
                scores = encoder.decode_scores(this_y=this_pred)

            # Get colors for center points according to responsible cell in the output grid cells.
            cells1 = encoder.get_cells(centers=centers_gt)
            cells2 = encoder.get_cells(centers=centers)
            colors1 = [Plotter._get_color(index=index) for index in np.ravel_multi_index(cells1.T, dims=cells_shape)]
            colors2 = [Plotter._get_color(index=index) for index in np.ravel_multi_index(cells2.T, dims=cells_shape)]

            # Draw bounding boxes and circles on the image.
            Plotter._draw_boxes(image=image, boxes=boxes_gt, colors=colors1, thickness=1)
            Plotter._draw_circles(image=image, points=centers_gt, colors=colors1, thickness=1, radius=12)
            if num_output_channels == 5:
                Plotter._draw_boxes(image=image, boxes=boxes, colors=colors2, thickness=2)
            Plotter._draw_circles(image=image, points=centers, colors=colors2, thickness=4, radius=4)
            Plotter._draw_scores(image=image, scores=scores, points=centers, colors=colors2, thickness=1)

            # Show the image with drawn bounding boxes and circles.
            cv2.imshow(window_name, image)

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
