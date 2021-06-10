import numpy as np
import cv2
import matplotlib.pyplot as plt

from parsers import KeypointsParser
from plotters import Skeleton, Drawer
from encoders import KeypointsEncoder


def main():

    # Instantiate a parser and encoder.
    parser = KeypointsParser(dataset="validation")
    encoder = KeypointsEncoder(input_shape=(256, 192), output_shape=(64, 48))

    # Create data generator.
    generator = (
        (
            cv2.imread(filename=parser.get_path(filename=filename)),
            np.array(box, dtype=np.float32),
            np.array(keypoints, dtype=np.float32).reshape(-1, 3)
        )
        for filename, box, keypoints in parser.info
    )

    # Show each image along with its bounding boxes in sequence.
    window_names = ["annotation", "input", "output"]
    for image, box_detection, keypoints in generator:

        # Expand detection box to get crop box.
        input_height, input_width = encoder.input_shape
        x1, y1, w, h = box_expanded = KeypointsEncoder.expand_box(
            box=box_detection,
            keypoints=keypoints,
            image_shape=image.shape[:2],
            aspect_ratio=input_width / input_height
        )
        print(f"AR={w / h:.2f}, [{x1:.1f}, {y1:.1f}, {w:.1f}, {h:.1f}], {image.shape[:2]}")

        # Cut the crop and move keypoints to crop coordinates.
        crop = image[int(y1):int(y1 + h), int(x1):int(x1 + w), ...]
        keypoints_moved = KeypointsEncoder.move_keypoints(
            origin=np.array([int(x1), int(y1)], dtype=np.float32),
            keypoints=keypoints
        )

        # Scale keypoints and resize the crop to input shape (can distort in edge case).
        crop_input = cv2.resize(crop, dsize=tuple(reversed(encoder.input_shape)))  # Use opencv convention (w, h).
        keypoints_input = KeypointsEncoder.scale_keypoints(
            keypoints=keypoints_moved,  # Keypoints in crop (expanded box) x, y.
            from_shape=(int(y1 + h) - int(y1), int(x1 + w) - int(x1)),  # Expanded box shape.
            to_shape=encoder.input_shape
        )

        # Encode keypoints into heatmap.
        heatmap = encoder.encode(keypoints=keypoints_input)

        # Scale keypoints and resize the crop to output shape.
        crop_output = cv2.resize(crop, dsize=tuple(reversed(encoder.output_shape)))  # Opencv convention (w, h).
        keypoints_output = encoder.scale_keypoints(
            keypoints=keypoints_input,
            from_shape=encoder.input_shape,
            to_shape=encoder.output_shape
        )

        # Draw skeletons with bounding boxes on the image and crop.
        Skeleton.draw(image=crop_input, keypoints=keypoints_input)
        Skeleton.draw(image=crop_output, keypoints=keypoints_output)
        Skeleton.draw(image=image, keypoints=keypoints, box=box_detection)
        Drawer.draw_boxes(image=image, boxes=box_expanded[np.newaxis, :], colors=[(255, 255, 255)])

        # Show the image with drawn bounding boxes.
        cv2.imshow(window_names[0], image)
        cv2.imshow(window_names[1], crop_input)
        cv2.imshow(window_names[2], crop_output)

        # Break the loop if key 'q' was pressed.
        if cv2.waitKey() & 0xFF == ord("q"):
            break

        # Show the output crop heatmap and the encoded and decoded keypoints.
        plt.figure()
        plt.imshow(cv2.cvtColor(crop_output, cv2.COLOR_BGRA2RGB))
        plt.imshow(heatmap.sum(axis=-1), alpha=0.7, interpolation="bilinear", cmap=plt.cm.get_cmap("viridis"))
        plt.colorbar()
        plt.show()

        # Check decode method.
        keypoints_decoded = encoder.decode(heatmap=heatmap, interpolate=True)
        is_keypoint_decoded = keypoints_decoded[:, 2] != 0
        is_keypoint_scaled = keypoints_input[:, 2] != 0
        print("Number of keypoints:", is_keypoint_decoded.sum(), is_keypoint_scaled.sum())
        print("Mean L2 error: ", np.mean(np.linalg.norm(keypoints_decoded[:, :2] - keypoints_input[:, :2], axis=-1)))
        print("Max L2 error: ", np.max(np.linalg.norm(keypoints_decoded[:, :2] - keypoints_input[:, :2], axis=-1)))

    # Close the windows.
    for window_name in window_names:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
