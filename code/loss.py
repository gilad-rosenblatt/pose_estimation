import numpy as np
import tensorflow as tf
from dataset import Dataset


# TODO make dtype adaptable.
# TODO use floats instead of numpy scalars for weight attributes.


class WeightedMSE(tf.keras.losses.Loss):
    """Custom MSE-based loss for object detection."""

    def __init__(self, weight_no_object=0, weight_object=1, name="weighted_mse"):
        """
        Initialize custom loss object.

        :param float weight_no_object: 0-1 weight to use for grid cells in which and object is not present.
        :param float weight_object: 0-1 weight to use for grid cells in which and object is present.
        :param str name: name of the loss object.
        """
        super().__init__(name=name)
        self.weight_no_object = np.array([weight_no_object], dtype=np.float32)
        self.weight_object = np.array([weight_object], dtype=np.float32)

    def call(self, y_true, y_pred):
        """
        Calculate weighted mean of squared errors between true and predicted y values.

        :param tf.Tensor y_true: tensor of ground-truth values.
        :param tf.Tensor y_pred: tensor of predicted values.
        :return tf.Tensor: loss value.
        """
        weights = tf.where(y_true == 1, self.weight_object, self.weight_no_object)
        return tf.reduce_mean(tf.multiply(weights, tf.square(tf.subtract(y_true, y_pred))))
        # return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))


def main():
    _, y = Dataset(batch_size=3, dataset="validation", output_type=0)[0]

    y_true = tf.convert_to_tensor(y * 1.0)
    y_pred = tf.convert_to_tensor(y * 0.8)

    loss = WeightedMSE(weight_object=5, weight_no_object=0.5)

    print(loss.call(y_true=y_true, y_pred=y_pred))


if __name__ == "__main__":
    main()
