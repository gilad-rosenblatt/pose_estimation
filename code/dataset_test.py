import unittest
import cv2
import numpy as np
from dataset import Dataset


class TestDataset(unittest.TestCase):
    """Test the Dataset class."""

    def test_encode_decode(self):
        """Test that decoding encoded boxes for a batch yields the original (scaled) box annotations for that batch."""

        # Load dataset generator object.
        dataset = Dataset(batch_size=64, dataset="train")

        # Select index of the batch to test.
        batch_index = 0

        # Get batch corresponding to input index using __getitem__.
        _, y = dataset[batch_index]

        # Get the batch annotation and path data.
        batch_info = dataset._parser.info[batch_index * dataset.batch_size: (batch_index + 1) * dataset.batch_size]

        # Test equality of annotation and encoded/decoded boxes for each image.
        for this_y, (filename, box_list) in zip(y, batch_info):

            # Extract image and detection bounding boxes.
            boxes_out = dataset._encoder.decode(this_y)

            # Load the image and boxes and resize to model input shape.
            _, boxes_in = dataset._resize(
                image=cv2.imread(dataset._parser.get_path(filename)),
                boxes=np.array(box_list, dtype=np.float32)
            )

            # Test output/input box equality for all input boxes other than those sharing the same cell (one per cell).
            # TODO Make sure all input boxes that are omitted share same cell with ones not omitted.
            if boxes_in.shape != boxes_out.shape:
                for box in boxes_out:
                    # Check if box contains in annotation boxes.
                    self.assertTrue(np.isclose(boxes_in, box).all(axis=1).any())
            else:
                np.testing.assert_array_almost_equal(np.sort(boxes_in, axis=0), np.sort(boxes_out, axis=0), decimal=4)
