import os

import numpy as np
import tensorflow as tf

from datasets import KeypointsDataset
from plotters import KeypointsPlotter


def main():
    # Load dataset and get first batch.
    ds_validation = KeypointsDataset(batch_size=64, dataset="validation")
    x, y_true = ds_validation[0]

    # Define models to show.
    filenames = dict(
        stage_1="my_model_tim1622826272.136246_bsz64_epo21_ckp01"  # 15ep@LR0.001&100ROP.
    )

    # Load models and show their predictions on the test batch (press "q" to exit).
    cls_threshold = 0.8
    nms_threshold = 0.3
    models_dir = os.path.join("..", "models", "keypoints")
    for stage, filename in filenames.items():
        print(f"Loading {stage}...")
        model = tf.keras.models.load_model(os.path.join(models_dir, filename), compile=False)
        y_prob = model.predict(x=x)
        # y_pred = np.where(y_prob[..., 0:1] > cls_threshold, y_prob, 0)  # Classify by applying threshold score.
        KeypointsPlotter.show_batch(x=x, y_true=y_true, y_pred=y_pred, nms_threshold=nms_threshold)


if __name__ == "__main__":
    main()
