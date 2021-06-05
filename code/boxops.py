import numpy as np


class IOU:
    """Intersection over union between bounding boxes."""

    @staticmethod
    def score(box1, box2):
        """
        Calculate the intersection over union between two arrays of boxes ("element-wise") keeping the same
        dimensions as the input. Boxes are in the last dimension in the format [x1, y1, w, h] (1: upper left corner).
        The boxes arrays need not have the same dimensions if they can be broadcast together into the same dimensions.

        :param np.ndarray box1: (..., 4) array of [x1, y1, w, h] boxes where x1, y1 is the upper left corner.
        :param np.ndarray box2: (..., 4) array of [x1, y1, w, h] boxes where x1, y1 is the upper left corner.
        :return np.ndarray: (..., 4) intersection over union score between boxes in arrays box1 and box2 "element-wise".
        """

        # Calculate intersection box upper left and lower right corners.
        x1 = np.maximum(box1[..., 0:1], box2[..., 0:1])
        y1 = np.maximum(box1[..., 1:2], box2[..., 1:2])
        x2 = np.minimum(box1[..., 0:1] + box1[..., 2:3], box2[..., 0:1] + box2[..., 2:3])
        y2 = np.minimum(box1[..., 1:2] + box1[..., 3:4], box2[..., 1:2] + box2[..., 3:4])

        # Extract intersection box width and height and clip at 0 if there is no intersection (h < 0 or w < 0).
        w = np.clip(a=x2 - x1, a_min=0, a_max=float("inf"))
        h = np.clip(a=y2 - y1, a_min=0, a_max=float("inf"))

        # Calculate the area of each box.
        area1 = box1[..., 2:3] * box1[..., 3:4]
        area2 = box2[..., 2:3] * box2[..., 3:4]

        # Calculate intersection and union areas.
        intersection = w * h
        union = area1 + area2 - intersection

        # Return intersection over union (add epsilon for numerical stabilization).
        return intersection / (union + 1e-12)


class NMS:
    """Non-maximum suppression on bounding boxes."""

    @staticmethod
    def perform(boxes, scores, threshold):
        """
        Perform non-maximum suppression (NMS) on an array of bounding boxes.

        :param np.ndarray boxes: (num_boxes, 4) array of [x1, y1, w, h] boxes where x1, y1 is the upper left corner.
        :param np.ndarray scores: (num_boxes, 1) corresponding array of confidence scores for the input bounding boxes.
        :param float threshold: the intersection over union threshold to use for non-maximum suppression.
        :return np.ndarray: new (num_keep_boxes, 4) array of [x1, y1, w, h] boxes where x1, y1 is the upper left corner.
        """

        # Sort boxes and their corresponding scores according to descending scores (not in-place, returns a copy).
        descending_score_order = np.argsort(scores, axis=0)[::-1, ...]
        scores = np.take_along_axis(scores, descending_score_order, axis=0)
        boxes = np.take_along_axis(boxes, descending_score_order, axis=0)

        # Instantiate a boolean indexing array for boxes to keep after NMS.
        keep = np.ones(shape=scores.shape, dtype=bool)

        # Iterate over boxes in ascending score order (no need to explicitly consider the last box).
        for this_index, (box, score) in enumerate(zip(boxes[:-1], scores[:-1])):

            # Skip this box if it has already been excluded.
            if not keep[this_index]:
                continue

            # Calculate IOU-scores between this box and all boxes with a lower confidence score (this_index + 1 < len).
            iou_scores = IOU.score(box, boxes[this_index + 1:])

            # Exclude all boxes with an above-threshold IOU score against this box (and a lower confidence score).
            keep[this_index + 1:][iou_scores > threshold] = False

        # Return un-excluded boxes and their corresponding confidence scores (in descending order).
        return np.compress(keep.ravel(), boxes, axis=0), np.compress(keep.ravel(), scores, axis=0)


if __name__ == "__main__":

    boxes = np.array([
        [0.5, 0.71, 0.20, 0.2],
        [0.5, 0.69, 0.20, 0.2],
        [0.8, 0.11, 0.60, 0.1],
        [0.8, 0.12, 0.63, 0.1],
        [0.8, 0.10, 0.61, 0.1],
        [0.8, 0.11, 0.60, 0.1],
        [0.1, 0.55, 0.60, 0.9]
    ])

    scores = np.array([
        [0.6],
        [0.4],
        [0.3],
        [0.3],
        [0.9],
        [0.7],
        [0.8]
    ])

    for box, score in zip(*NMS.perform(boxes, scores, threshold=0.5)):
        print(box, score)
