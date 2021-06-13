# TODO Define custom TF metric class?
# TODO deeper/more/better blocks
# TODO add anchors.
# TODO go multiclass.
# TODO data augmentation (e.g., train on several scales).
# TODO create a README file per saved model with hyperparameters.
# TODO transition into instance segmentation?
# TODO make it into YOLOv5-ish

import os

import cv2
import numpy as np
import tensorflow as tf

from datasets import KeypointsDataset
from encoders import KeypointsEncoder
from plotters import Skeleton


def video_test(filename, show=False):

    # Collect the (first-only) video filename in the test folder.
    test_dir = os.path.join("..", "data", "my_test", "keypoints", "videos")
    filenames = [f for f in os.listdir(test_dir)]
    print(f"Available files video in test folder: {' '.join(filenames)}.")
    assert filename in filenames
    this_filename = filename

    # Load the model.
    models_dir = os.path.join("..", "models", "keypoints")
    model_filename = "my_model_tim1623386780.9279761_bsz64_epo15"
    model = tf.keras.models.load_model(os.path.join(models_dir, model_filename), compile=False)

    # Load the encoder to read output heatmaps into keypoints.
    encoder = KeypointsEncoder(
        input_shape=KeypointsDataset.INPUT_SHAPE,
        output_shape=KeypointsDataset.OUTPUT_SHAPE
    )

    # Start a video capture.
    full_filename = os.path.join(test_dir, this_filename)
    capture = cv2.VideoCapture(full_filename)
    print(f"Opened video capture for {full_filename}.")

    # Open a video writer.
    pred_dir = os.path.join("..", "data", "predictions", "keypoints", "videos")
    new_full_filename = os.path.join(pred_dir, f"{this_filename[:-4]}_pred.avi")  # Remove 4-char extension (".mp4).
    four_cc_code = cv2.VideoWriter_fourcc(*"XVID")
    frames_per_second = int(capture.get(cv2.CAP_PROP_FPS))
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))  # (h, w).
    video_writer = cv2.VideoWriter(
        new_full_filename,
        four_cc_code,
        frames_per_second,
        tuple(reversed(frame_size))  # Opencv convention (width, height).
    )
    print(f"Opened video writer to save to {new_full_filename}.")

    # Iterate on video frames.
    window_name = "video_test"
    frame_num = 0
    while True:

        # Get next frame or break if not retrieved (e.g., video ended) and save its height and width.
        retrieved, this_image = capture.read()
        if not retrieved:
            print(f"Done after {frame_num} frames.")
            break
        *image_shape, _ = this_image.shape

        # Resize to model input shape and normalize to 0-1.
        resized_image = cv2.resize(this_image, dsize=tuple(reversed(KeypointsDataset.INPUT_SHAPE)))  # Opencv (w, h).
        this_x = resized_image / 255

        # Predict on this image and decode keypoints.
        y_prob = model.predict(x=this_x[np.newaxis, ...])
        this_pred = y_prob[0, ...]
        keypoints_input = encoder.decode(
            heatmap=this_pred,
            interpolate=True,
            threshold=0.3
        )

        # Scale keypoints to original image dimensions.
        keypoints = KeypointsEncoder.scale_keypoints(
            keypoints=keypoints_input,
            from_shape=KeypointsDataset.INPUT_SHAPE,
            to_shape=image_shape
        )

        # Draw skeleton on this image.
        Skeleton.draw(
            image=this_image,
            keypoints=keypoints,
            thickness=4,
            radius=5
        )

        # Save this image (with drawn skeleton prediction).
        video_writer.write(this_image)

        # Show the image with drawn bounding box (with option to break by pressing "q").
        if show:
            cv2.imshow(window_name, this_image)
            if cv2.waitKey() & 0xFF == ord("s"):
                break

        # Increment frame number.
        frame_num += 1

    # Close window.
    capture.release()
    video_writer.release()
    if show:
        cv2.destroyWindow(window_name)


def image_test(show=False):

    # Collect all filenames for images in the test folder (only "jpg" extension).
    test_dir = os.path.join("..", "data", "my_test", "keypoints", "images")
    filenames = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith("jpg")]

    # Load the model.
    models_dir = os.path.join("..", "models", "keypoints")
    model_filename = "my_model_tim1623386780.9279761_bsz64_epo15"
    model = tf.keras.models.load_model(os.path.join(models_dir, model_filename), compile=False)

    # Load the encoder to read output heatmaps into keypoints.
    encoder = KeypointsEncoder(
        input_shape=KeypointsDataset.INPUT_SHAPE,
        output_shape=KeypointsDataset.OUTPUT_SHAPE
    )

    # Iterate over files in folder and predict & save on each image.
    window_name = "test"
    for this_filename in filenames:

        # Load this image and save its height and width.
        this_image = cv2.imread(os.path.join(test_dir, this_filename))
        *image_shape, _ = this_image.shape

        # Resize to model input shape and normalize to 0-1.
        resized_image = cv2.resize(this_image, dsize=tuple(reversed(KeypointsDataset.INPUT_SHAPE)))  # Opencv (w, h).
        this_x = resized_image / 255

        # Predict on this image and decode keypoints.
        y_prob = model.predict(x=this_x[np.newaxis, ...])
        this_pred = y_prob[0, ...]
        keypoints_input = encoder.decode(
            heatmap=this_pred,
            interpolate=True,
            threshold=0.1
        )

        # Scale keypoints to original image dimensions.
        keypoints = KeypointsEncoder.scale_keypoints(
            keypoints=keypoints_input,
            from_shape=KeypointsDataset.INPUT_SHAPE,
            to_shape=image_shape
        )

        # Draw skeleton on this image.
        Skeleton.draw(
            image=this_image,
            keypoints=keypoints,
            thickness=4,
            radius=5
        )

        # Save this image (with drawn skeleton prediction).
        pred_dir = os.path.join("..", "data", "predictions", "keypoints", "images")
        cv2.imwrite(os.path.join(pred_dir, f"{this_filename[:-4]}_pred.jpg"), this_image)  # Remove 4-char extension.

        # Show the image with drawn bounding box (with option to break by pressing "q").
        if show:
            cv2.imshow(window_name, this_image)
            if cv2.waitKey() & 0xFF == ord("s"):
                break

    # Close window.
    if show:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    video_test(show=False, filename="moves221.mp4")
    # image_test(show=False)
