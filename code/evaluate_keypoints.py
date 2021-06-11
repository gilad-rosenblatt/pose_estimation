import os

import tensorflow as tf

from datasets import KeypointsDataset
from plotters import KeypointsPlotter
from models import get_keypoints_detection_model


def get_model_from_checkpoint(epoch, *args, **kwargs):
    checkpoint_path = os.path.join("..", "training", "keypoints", f"cp-{epoch:03d}.ckpt")
    return get_keypoints_detection_model(*args, **kwargs).load_weights(checkpoint_path)


def main():
    # Load dataset and get first batch.
    ds_validation = KeypointsDataset(batch_size=64, dataset="validation")
    x, y_true = ds_validation[0]

    # Define models to show.
    filenames = dict(
        # stage_1="my_model_tim1623386780.9279761_bsz64_epo15",  # 15ep@LR0.001&100ROP.
        stage_2="my_model_tim1623401683.0648534_bsz64_epo40"  # 40ep@LR0.001&1000ROP.
    )

    # Load models and show their predictions on the test batch (press "q" to exit).
    models_dir = os.path.join("..", "models", "keypoints")
    for stage, filename in filenames.items():
        print(f"Loading {stage}...")
        model = tf.keras.models.load_model(os.path.join(models_dir, filename), compile=False)
        y_prob = model.predict(x=x)
        KeypointsPlotter.show_batch(x=x, y_true=y_true, y_pred=y_prob, show_keypoints=True)


if __name__ == "__main__":
    main()
