import numpy as np
import tensorflow as tf
from dataset import Dataset
from plotters import Plotter
from model import get_model


def main():
    # Load dataset and get first batch.
    ds_validation = Dataset(batch_size=32, dataset="validation")
    x, y_true = ds_validation[0]

    # Get stage 1 model.
    # About compile=False: <https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1>.
    model = tf.keras.models.load_model("../models/my_model_tim1622613991.4143317_bsz64_epo10", compile=False)

    # Predict on first batch and show results.
    y_prob = model.predict(x=x)
    y_pred = np.where(y_prob[..., 0:1] > 0.9, y_prob, 0)  # Nullify front channel where score is smaller than threshold.
    Plotter.show_batch(x=x, y_true=y_true, y_pred=y_pred)

    # Get stage 2 model.
    model = get_model()
    model.load_weights("../training/cp-006-0.07.ckpt")

    # Predict on first batch and show results.
    y_prob = model.predict(x=x)
    y_pred = np.where(y_prob[..., 0:1] > 0.8, y_prob, 0)
    Plotter.show_batch(x=x, y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    main()
