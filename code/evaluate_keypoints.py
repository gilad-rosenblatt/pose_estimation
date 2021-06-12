import os
from time import time

import tensorflow as tf

from datasets import KeypointsDataset
from models import get_keypoints_detection_model
from plotters import KeypointsPlotter


def save_model_from_checkpoint(epoch, batch_size, base_model_filename=None, *args, **kwargs):
    checkpoint_path = os.path.join("..", "training", "keypoints", f"cp-{epoch:03d}.ckpt")
    models_dir = os.path.join("..", "models", "keypoints")
    if base_model_filename:
        model = tf.keras.models.load_model(os.path.join(models_dir, base_model_filename), compile=False)
    else:
        model = get_keypoints_detection_model(*args, **kwargs)
    model.load_weights(checkpoint_path)
    model_path = os.path.join(models_dir, "my_model")
    model.save(f"{model_path}_tim{time()}_bsz{batch_size}_epo{epoch}")


def main():
    # Load dataset and get first batch.
    ds_validation = KeypointsDataset(batch_size=64, dataset="validation")
    x, y_true = ds_validation[0]

    # Define models to show.
    filenames = dict(
        stage_1="my_model_tim1623386780.9279761_bsz64_epo15",  # 15ep@LR0.001&100ROP.  MSE all channels. No overfit.
        # stage_2="my_model_tim1623401683.0648534_bsz64_epo40",  # 40ep@LR0.001&1000ROP. KP Y/N loss. Overfit.
        # stage_3="my_model_tim1623423664.7099812_bsz64_epo7",  # 40ep@LR0.001&1000ROP. Disk mask loss. BAD.
        # stage_4="my_model_junk_tim1623438554.4141183_bsz64_epo20",  # 20ep@LR0.001&1000ROP. KP Y/N loss. Bit overfit.
        # stage_5="my_model_tim1623504195.0734112_bsz64_epo13"  # 13ep@LR0.001&1000ROP. KP Y/N loss. Best ckp @ stage 4.
    )

    # Load models and show their predictions on the test batch (press "q" to exit).
    models_dir = os.path.join("..", "models", "keypoints")
    for stage, filename in filenames.items():
        print(f"Loading {stage}...")
        model = tf.keras.models.load_model(os.path.join(models_dir, filename), compile=False)
        y_prob = model.predict(x=x)
        KeypointsPlotter.show_batch(
            x=x,
            y_true=y_true,
            y_pred=y_prob,
            show_keypoints=True,
            interpolate=True,
            threshold=0.1
        )


if __name__ == "__main__":
    # save_model_from_checkpoint(
    #     epoch=13,
    #     batch_size=64,
    #     base_model_filename="my_model_junk_tim1623438554.4141183_bsz64_epo20"
    # )
    main()
