import tensorflow as tf


# TODO add batch norm and dropout.
# TODO use a better activation (leaky relu or mish).

def get_model(out_channels=5, final_activation=tf.keras.activations.sigmoid):
    """:return tf.keras.Sequential: tensorflow fully-convolutional model."""
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, None, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding="same"),
        tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), activation=final_activation, padding="same"),
    ])
    return conv_model


if __name__ == "__main__":
    model = get_model()
    model.summary()
