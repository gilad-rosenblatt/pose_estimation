import tensorflow as tf


# TODO add batch norm and dropout.
# TODO use a better activation (leaky relu or mish).

def get_basic_model():
    """:return tf.keras.Sequential: tensorflow fully-convolutional model for one-step object detection."""
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
    model = get_basic_model()
    model.summary()
