import cv2
import numpy as np
from utils import BoxEncoder


class Plotter:

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
        for this_x, this_y in zip(x, y_true):

            # Load the image and instantiate a numpy array of boxes (one per row)..
            image = (this_x * 255).astype(np.uint8)
            boxes = encoder.decode(this_y=this_y)

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
