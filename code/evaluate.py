import os

import numpy as np
import tensorflow as tf

from datasets import DetectionsDataset
from plotters import DetectionsPlotter


def main():
    # Load dataset and get first batch.
    ds_validation = DetectionsDataset(batch_size=64, dataset="validation")
    x, y_true = ds_validation[0]

    # Define models to show.
    filenames = dict(
        # stage_1="my_model_tim1622613991.4143317_bsz64_epo10",  # Plotter will raise dimension mismatch exception.
        # stage_2="my_model_tim1622657298.7396855_bsz64_epo10",  # Plotter will raise dimension mismatch exception.
        # stage_3="my_model_tim1622707835.3077517_bsz64_epo15_ckp10",
        # stage_3="my_model_tim1622799333.2177446_bsz64_epo15_ckp10",  # 15ep@LR0.0001/100ROP.
        # stage_4="my_model_tim1622744262.2669034_bsz64_epo15_ckp08",  # 15ep@LR0.0001/100ROP.
        # stage_4="my_model_tim1622822245.2772586_bsz64_epo21",  # 15ep@LR0.001&100ROP+6ep@0.00001&100ROP
        stage_4="my_model_tim1622826272.136246_bsz64_epo21_ckp01"  # 15ep@LR0.001&100ROP+5ep@0.00001&100ROP+1ep@0.00001.
    )

    # Load models and show their predictions on the test batch (press "q" to exit).
    # About compile=False: <https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1>.
    cls_threshold = 0.8
    nms_threshold = 0.3
    models_dir = "../models"
    for stage, filename in filenames.items():
        print(f"Loading {stage}...")
        model = tf.keras.models.load_model(os.path.join(models_dir, filename), compile=False)
        y_prob = model.predict(x=x)
        y_pred = np.where(y_prob[..., 0:1] > cls_threshold, y_prob, 0)  # Classify by applying threshold score.
        DetectionsPlotter.show_batch(x=x, y_true=y_true, y_pred=y_pred, nms_threshold=nms_threshold)


if __name__ == "__main__":
    main()
