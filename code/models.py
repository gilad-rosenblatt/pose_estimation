import tensorflow as tf


def get_model(batch_normalization=False, activation="relu", dropout=False):
    """
    Build a fully-convolutional one-step object detection model based on 6x6 cell cell detection with 1 box per cell.
    The model including bach normalization and dropout layers, as well as a leaky ReLU activation.

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


def get_basic_model():
    """
    Build a fully-convolutional one-step object detection model based on 6x6 cell cell detection with 1 box per cell.

    :return tf.keras.Sequential: tensorflow model for one-step object detection.
    """
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation=tf.keras.activations.relu, padding="same")(x)
    out1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation=tf.keras.activations.sigmoid, padding="same")(x)
    out2 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation=tf.keras.activations.linear, padding="same")(x)
    outputs = tf.keras.layers.Concatenate()([out1, out2])
    return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = get_model()
    model.summary()
