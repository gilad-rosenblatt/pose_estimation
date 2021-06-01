import cv2
import numpy as np
from utils import BoxEncoder


class Plotter:
    """Plotter for images with bounding boxes."""

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
        encoder = BoxEncoder(image_shape=image_shape, cells_shape=cells_shape)

        # Show each image along with its bounding boxes in sequence.
        window_name = "annotation_boxes"
        for this_x, this_y, this_pred in zip(x, y_true, y_pred):

            # Cast image to uint8.
            image = (this_x * 255).astype(np.uint8)

            # Decode ground-truth bounding boxes and cell center points for predictions.
            boxes = encoder.decode(this_y=this_y)
            points = encoder.decode_centers(this_y=this_pred)

            # Draw bounding boxes and circles on the image.
            Plotter._draw_boxes(image=image, boxes=boxes)
            Plotter._draw_circles(image=image, points=points)

            # Show the image with drawn bounding boxes and circles.
            cv2.imshow(window_name, image)

            # Break the loop if key 'q' was pressed.
            if cv2.waitKey() & 0xFF == ord("q"):
                break

        # Close the window.
        cv2.destroyWindow(window_name)

    @staticmethod
    def _draw_boxes(image, boxes):
        """
        Draw boxes one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw boxes on (assumed uint8 BGR image).
        :param np.ndarray boxes: boxes array sized [num_boxes, x1, y1, width, height] in image pixel units.
        """
        colors = 255 * np.random.rand(32, 3)
        for index, (x1, y1, w, h) in enumerate(boxes):
            image = cv2.rectangle(
                image,
                (int(x1), int(y1)),  # Upper left corner.
                (int(x1 + w), int(y1 + h)),  # Lower right corner.
                color=colors[index % 32, :],
                thickness=2
            )

    @staticmethod
    def _draw_circles(image, points):
        """
        Draw circles one-by-one on the image (carries side effect to image array).

        :param np.ndarray image: image to draw boxes on (assumed uint8 BGR image).
        :param np.ndarray points: points array sized [num_points, xc, yc] in image pixel units.
        """
        colors = 255 * np.random.rand(32, 3)
        for index, (xc, yc) in enumerate(points):
            image = cv2.circle(
                image,
                center=(int(xc), int(yc)),  # Center point.
                radius=12,
                color=colors[index % 32, :],
                thickness=2
            )
