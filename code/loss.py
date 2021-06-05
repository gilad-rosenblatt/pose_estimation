import tensorflow as tf

from dataset import Dataset


class WeightedMSE(tf.keras.losses.Loss):
    """Simple MSE-based loss for object detection."""

    def __init__(self, weight_noo=0.5, weight_obj=1, name="weighted_mse"):
        """
        Initialize custom loss object.

        :param float weight_noo: 0-1 weight to use for classification in cells in which an object is not present.
        :param float weight_obj: 0-1 weight to use for classification in cells in which an object is present.
        :param str name: name of the loss object.
        """
        super().__init__(name=name)
        self.weight_noo = tf.constant([weight_noo], dtype=tf.float32)
        self.weight_obj = tf.constant([weight_obj], dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Calculate weighted mean of squared errors between true and predicted y values.

        :param tf.Tensor y_true: tensor of ground-truth values.
        :param tf.Tensor y_pred: tensor of predicted values.
        :return tf.Tensor: loss value.
        """
        weights = tf.where(y_true[..., 0:1] == 1, self.weight_obj, self.weight_noo)
        mse_cls = tf.reduce_mean(tf.multiply(weights, tf.square(tf.subtract(y_true, y_pred))))
        return mse_cls


class DetectionLoss(WeightedMSE):
    """Compound MSE-based loss for object detection containing classification and regression parts."""

    def __init__(self, weight_box=5, *args, **kwargs):
        """
        Initialize custom detection loss object.

        :param float weight_box: 0-1 weight to use for regression of box params in cells in which an object is present.
        """
        super().__init__(*args, **kwargs)
        self.weight_box = tf.constant([weight_box], dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Calculate weighted mean of squared errors between true and predicted y values separately for classification
        and regression parts.

        :param tf.Tensor y_true: tensor of ground-truth values of shape (..., >1).
        :param tf.Tensor y_pred: tensor of predicted values of shape (..., >1)..
        :return tf.Tensor: loss value.
        """
        mse_cls = super().call(y_true=y_true[..., 0:1], y_pred=y_pred[..., 0:1])
        mask = tf.where(y_true[..., 0:1] == 1, self.weight_box, 0)
        mse_reg = tf.reduce_mean(tf.multiply(mask, tf.square(tf.subtract(y_true[..., 1:], y_pred[..., 1:]))))
        return mse_cls + mse_reg


class ScaledDetectionLoss(DetectionLoss):
    """Compound MSE-based loss for object detection containing classification and two (scaled) regression parts."""

    def __init__(self, *args, **kwargs):
        """Initialize custom detection loss object."""
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        """
        Calculate weighted mean of squared errors between true and predicted y values separately for classification
        and regression parts, and square the width and height dimensions to "resist" box scale changes.

        :param tf.Tensor y_true: tensor of ground-truth values of shape (..., >1).
        :param tf.Tensor y_pred: tensor of predicted values of shape (..., >1)..
        :return tf.Tensor: loss value.
        """
        mse1 = super().call(y_true=y_true[..., 0:3], y_pred=y_pred[..., 0:3])
        mask = tf.where(y_true[..., 0:1] == 1, self.weight_box, 0)
        mse2 = tf.reduce_mean(tf.multiply(mask, tf.square(tf.subtract(
            tf.sqrt(tf.abs(y_true[..., 3:])),
            tf.sqrt(tf.abs(y_pred[..., 3:]))
        ))))
        return mse1 + mse2


if __name__ == "__main__":
    _, y = Dataset(batch_size=64, dataset="validation")[0]
    y_true = tf.convert_to_tensor(y * 1.0)
    y_pred = tf.convert_to_tensor(y * 0.8)
    print(ScaledDetectionLoss().call(y_true=y_true, y_pred=y_pred))
