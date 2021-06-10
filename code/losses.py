import tensorflow as tf

from datasets import DetectionsDataset, KeypointsDataset


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


class KeypointsLoss(tf.keras.losses.Loss):

    def __init__(self, weight_nok=0.1, weight_kps=100, name="keypoints_mse"):
        """
        Initialize custom loss object.

        :param float weight_nok: 0-1 weight to use for pixels in which a keypoint is not present.
        :param float weight_kps: 0-1 weight to use for pixels in which a keypoint is present.
        :param str name: name of the loss object.
        """
        super().__init__(name=name)
        self.weight_nok = tf.constant([weight_nok], dtype=tf.float32)
        self.weight_kps = tf.constant([weight_kps], dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Calculate weighted mean of squared errors between true and predicted y values.

        :param tf.Tensor y_true: tensor of ground-truth values.
        :param tf.Tensor y_pred: tensor of predicted values.
        :return tf.Tensor: loss value.
        """
        weights = tf.where(y_true[...] > 0.05, self.weight_kps, self.weight_nok)
        weights = tf.where(tf.math.count_nonzero(y_true, [1, 2], keepdims=True) > 0, weights, 0.0)
        return tf.reduce_mean(tf.multiply(weights, tf.square(tf.subtract(y_true, y_pred))))


if __name__ == "__main__":
    _, y = KeypointsDataset(batch_size=64, dataset="validation")[0]
    y_true = tf.convert_to_tensor(y * 1.0)
    y_pred = tf.convert_to_tensor(y * 0.2)
    print(KeypointsLoss().call(y_true=y_true, y_pred=y_pred))
