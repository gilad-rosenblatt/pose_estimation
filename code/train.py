import tensorflow as tf
from dataset import Dataset
from model import get_model
from loss import WeightedMSE


# TODO define first on a small dataset and see overfit (changes in Dataset needed).
# TODO how to evaluate the model?! Define custom metric?
# TODO create a README file per saved model with hyperparameters and make folder per model in models.


def main():
    # Load dataset generators.
    batch_size = 64
    ds_train = Dataset(batch_size=batch_size, dataset="validation", output_type=0)
    ds_validation = Dataset(batch_size=batch_size, dataset="validation", output_type=0)

    # Get Model.
    model = get_model(out_channels=1)
    model.summary()

    # Define optimizer.
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    # Define loss.
    weighted_mse = WeightedMSE(
        weight_object=5,
        weight_no_object=0.5
    )

    # Compile the model and optimizer.
    model.compile(
        optimizer=optimizer,
        loss=weighted_mse,
        metrics=["mse", "acc"]
    )

    # Train the model w/callbacks.
    epochs = 2
    model.fit(
        x=ds_train,
        validation_data=ds_validation,
        epochs=epochs
    )


if __name__ == "__main__":
    main()
