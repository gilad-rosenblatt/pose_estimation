import tensorflow as tf


def get_object_detection_model(batch_normalization=False, activation="relu", dropout=False):
    """
    Build a fully-convolutional one-step object detection model based on 6x6 cell cell detection with 1 box per cell.
    The model optionally (not by default) allows bach normalization, dropout layers and a leaky ReLU activation.

    :return tf.keras.Sequential: tensorflow model for one-step object detection.
    """

    # Define the basic building block of the model.
    def block(x, repetitions, filters, pooling=True):
        for _ in range(repetitions):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation=None, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x) if batch_normalization else x
            x = tf.keras.layers.ReLU()(x) if activation == "relu" else tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return tf.keras.layers.MaxPooling2D()(x) if pooling else x

    # Define the model backbone.
    x = inputs = tf.keras.layers.Input(shape=(None, None, 3))
    x = block(x, repetitions=2, filters=32, pooling=True)
    x = block(x, repetitions=3, filters=64, pooling=True)
    x = block(x, repetitions=4, filters=128, pooling=True)
    x = block(x, repetitions=3, filters=256, pooling=True)
    x = block(x, repetitions=3, filters=512, pooling=True)
    x = block(x, repetitions=3, filters=1024, pooling=False)
    x = tf.keras.layers.Dropout(rate=0.3)(x) if dropout else x

    # Define the model classification + regression head and return model.
    out1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation=tf.keras.activations.sigmoid, padding="same")(x)
    out2 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation=tf.keras.activations.linear, padding="same")(x)
    outputs = tf.keras.layers.Concatenate()([out1, out2])
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_keypoints_detection_model(batch_normalization=False, activation="relu", dropout=False):
    """
    Build a fully-convolutional one-step person keypoint detection model based on a 17-keypoint heatmap encoding.
    The model optionally (not by default) allows bach normalization, dropout layers and a leaky ReLU activation.

    :return tf.keras.Sequential: tensorflow model for one-step person keypoints detection.
    """

    # Number of keypoints to encode.
    num_keypoints = 17

    # Define the basic building block of the encoder.
    def encoder_block(x, repetitions, filters, pooling=True):
        for _ in range(repetitions):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation=None, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x) if batch_normalization else x
            x = tf.keras.layers.ReLU()(x) if activation == "relu" else tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return tf.keras.layers.MaxPooling2D()(x) if pooling else x

    # Define the basic building block of the decoder.
    def decoder_block(x, filters):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding="same")(x)  # No act.
        x = tf.keras.layers.BatchNormalization()(x) if batch_normalization else x
        return tf.keras.layers.ReLU()(x)

    # Define the model encoder (detection backbone).
    x = inputs = tf.keras.layers.Input(shape=(None, None, 3))
    x = encoder_block(x, repetitions=2, filters=32, pooling=True)
    x = encoder_block(x, repetitions=3, filters=64, pooling=True)
    x = encoder_block(x, repetitions=4, filters=128, pooling=True)
    x = encoder_block(x, repetitions=3, filters=256, pooling=True)
    x = encoder_block(x, repetitions=3, filters=512, pooling=True)
    x = encoder_block(x, repetitions=3, filters=1024, pooling=False)
    x = tf.keras.layers.Dropout(rate=0.3)(x) if dropout else x

    # Define the model decoder (keypoints feature extractor).
    x = decoder_block(x, filters=256)
    x = decoder_block(x, filters=256)
    x = decoder_block(x, filters=256)

    # Define the model keypoint heatmap regression head and return model.
    outputs = tf.keras.layers.Conv2D(filters=num_keypoints, kernel_size=1, activation="sigmoid", padding="same")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = get_keypoints_detection_model()
    model.summary()
