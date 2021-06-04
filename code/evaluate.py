import os
import numpy as np
import tensorflow as tf
from dataset import Dataset
from plotters import Plotter


def main():
    # Load dataset and get first batch.
    ds_validation = Dataset(batch_size=32, dataset="validation")
    x, y_true = ds_validation[0]

    # Define models to show.
    filenames = dict(
        # stage_1="my_model_tim1622613991.4143317_bsz64_epo10",
        # stage_2="my_model_tim1622657298.7396855_bsz64_epo10",
        # stage_3="my_model_tim1622707835.3077517_bsz64_epo15_ckp10", # 1st session.
        stage_3="my_model_tim1622799333.2177446_bsz64_epo15_ckp10",  # 2nd session.
        # stage_4="my_model_tim1622744262.2669034_bsz64_epo15_ckp08",
    )

    # Load models and show their predictions on the test batch (press "q" to exit).
    # About compile=False: <https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1>.
    models_dir = "../models"
    for stage, filename in filenames.items():
        print(f"Loading {stage}...")
        model = tf.keras.models.load_model(os.path.join(models_dir, filename), compile=False)
        y_prob = model.predict(x=x)
        y_pred = np.where(y_prob[..., 0:1] > 0.85, y_prob, 0)  # Nullify front channel where score < threshold.
        Plotter.show_batch(x=x, y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    main()
