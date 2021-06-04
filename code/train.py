import tensorflow as tf
from time import time
from dataset import Dataset
from models import get_model
from loss import DetectionLoss
from tensorflow.python.keras.callbacks import TensorBoard


# TODO how to evaluate the model?! Define custom metric?
# TODO create a README file per saved model with hyperparameters and make folder per model in models.


def main():
    # Load dataset generators.
    batch_size = 64
    ds_train = Dataset(batch_size=batch_size, dataset="train")
    ds_validation = Dataset(batch_size=batch_size, dataset="validation")

    # Get Model.
    model = get_model(batch_normalization=True, activation="leaky_relu", dropout=True)
    model.summary()

    # Define optimizer.
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    # Define loss.
    weighted_mse = DetectionLoss(
        weight_obj=5,
        weight_noo=0.5,
        weight_box=5
    )

    # Compile the model and optimizer.
    model.compile(
        optimizer=optimizer,
        loss=weighted_mse,
        metrics=["mse"]
    )

    # Create callback to save model weights every epoch and reduce learning rate on plateau.
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="../training/cp-{epoch:03d}.ckpt",
        verbose=1,
        save_weights_only=True,
        save_best_only=True  # Latest checkpoint should be best.
    )
    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=learning_rate / 100
    )
    tensorboard = TensorBoard(f"../logs/{time()}")
    callbacks = [save_checkpoint, reduce_learning_rate, tensorboard]

    # Train the model w/callbacks.
    epochs = 15
    model.fit(
        x=ds_train,
        validation_data=ds_validation,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save the model.
    model_path = "../models/my_model"
    model.save(
        f"{model_path}_"
        f"tim{time()}_"
        f"bsz{batch_size}_"
        f"epo{epochs}"
    )


if __name__ == "__main__":
    main()
