import tensorflow as tf
from time import time
from dataset import Dataset
from model import get_model
from loss import WeightedMSE, DetectionLoss, ScaledDetectionLoss
from tensorflow.python.keras.callbacks import TensorBoard


# TODO how to evaluate the model?! Define custom metric?
# TODO create a README file per saved model with hyperparameters and make folder per model in models.


def main():
    # Load dataset generators.
    batch_size = 64
    ds_train = Dataset(batch_size=batch_size, dataset="train", output_type=1)
    ds_validation = Dataset(batch_size=batch_size, dataset="validation", output_type=1)

    # Get Model.
    model = get_model()
    model.summary()

    # Define optimizer.
    learning_rate = 0.0001
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
        metrics=["mse", "acc"]
    )

    # Create callback to save model weights every epoch and reduce learning rate on plateau.
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="../training/cp-{epoch:03d}-{val_loss:.2f}.ckpt",
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
    epochs = 10
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
