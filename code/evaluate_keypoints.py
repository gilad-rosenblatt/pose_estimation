import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datasets import KeypointsDataset
from plotters import KeypointsPlotter
from models import get_keypoints_detection_model
from encoders import KeypointsEncoder


def main():
    # Load dataset and get first batch.
    ds_validation = KeypointsDataset(batch_size=64, dataset="validation")
    x, y_true = ds_validation[0]

    encoder = KeypointsEncoder(input_shape=(256, 192), output_shape=(64, 48))

    # Define models to show.
    filenames = dict(
        stage_1="my_model_tim1622826272.136246_bsz64_epo21_ckp01"  # 15ep@LR0.001&100ROP.
    )

    # Load models and show their predictions on the test batch (press "q" to exit).
    models_dir = os.path.join("..", "models", "keypoints")
    for stage, filename in filenames.items():
        print(f"Loading {stage}...")
        # model = get_keypoints_detection_model(batch_normalization=True, activation="leaky_relu", dropout=True)
        # model.load_weights(os.path.join("..", "training", "keypoints", "cp-004.ckpt"))
        # model = tf.keras.models.load_model(os.path.join(models_dir, filename), compile=False)
        # y_prob = model.predict(x=x)
        # KeypointsPlotter.show_batch(x=x, y_true=y_true, y_pred=y_prob)
        for this_x, this_y in zip(x, y_true):
            # keypoints_decoded_true = encoder.decode(heatmap=this_y, interpolate=False)
            # keypoints_decoded_pred = encoder.decode(heatmap=this_prob, interpolate=False)
            # is_keypoint_decoded = keypoints_decoded[:, 2] != 0

            image = (this_x * 255).astype(np.uint8)
            plt.figure()
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
            # plt.imshow(this_y.sum(axis=-1), alpha=0.7, interpolation="bilinear", cmap=plt.cm.get_cmap("viridis"))
            plt.show()


if __name__ == "__main__":
    main()
