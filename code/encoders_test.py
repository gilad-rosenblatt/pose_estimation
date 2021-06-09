import numpy as np
import cv2
import matplotlib.pyplot as plt

from parsers import KeypointsParser
from plotters import Skeleton, Drawer
from encoders import KeypointsEncoder


def main():

    # Instantiate a parser and encoder.
    parser = KeypointsParser(dataset="validation")
    encoder = KeypointsEncoder(input_shape=(256, 192), output_shape=(128, 96))

    # Create data generator.
    generator = (
        (cv2.imread(filename=parser.get_path(filename=filename)), np.array(box), np.array(keypoints).reshape(-1, 3))
        for filename, box, keypoints in parser.info
    )

    # Show each image along with its bounding boxes in sequence.
    window_names = ["annotation", "input", "output"]
    for image, box_detection, keypoints in generator:

        # Expand the detection box to get the crop box.
        heatmap, box_crop = encoder.encode(box=box_detection, keypoints=keypoints, image_shape=image.shape[:2])
        x1, y1, w, h = box_crop
        print(f"AR={w / h:.2f}, [{x1:.1f}, {y1:.1f}, {w:.1f}, {h:.1f}], {image.shape[:2]}")

        # Move keypoints and cut the crop.
        keypoints_moved = encoder.move_keypoints(origin=box_crop[:2], keypoints=keypoints)
        crop = image[int(y1):int(y1 + h), int(x1):int(x1 + w), ...]
        crop_shape = crop.shape[:2]

        # Scale keypoints and resize the crop to input shape.
        keypoints_scaled = encoder.scale_keypoints(keypoints=keypoints_moved, from_shape=crop_shape, to_shape=encoder.input_shape)
        crop_scaled = cv2.resize(crop, dsize=tuple(reversed(encoder.input_shape)))  # Opencv convention (w, h).

        # Scale keypoints and resize the crop to output shape.
        keypoints_output = encoder.scale_keypoints(keypoints=keypoints_moved, from_shape=crop_shape, to_shape=encoder.output_shape)
        crop_output = cv2.resize(crop, dsize=tuple(reversed(encoder.output_shape)))  # Opencv convention (w, h).

        # Draw skeletons with bounding boxes on the image and crop.
        Skeleton.draw(image=crop_scaled, keypoints=keypoints_scaled)
        Skeleton.draw(image=crop_output, keypoints=keypoints_output)
        Skeleton.draw(image=image, keypoints=keypoints, box=box_detection)
        Drawer.draw_boxes(image=image, boxes=box_crop[np.newaxis, :], colors=[(255, 255, 255)])

        # Show the image with drawn bounding boxes.
        cv2.imshow(window_names[0], image)
        cv2.imshow(window_names[1], crop_scaled)
        cv2.imshow(window_names[2], crop_output)

        # Break the loop if key 'q' was pressed.
        if cv2.waitKey() & 0xFF == ord("q"):
            break

        # Show the output crop heatmap and the encoded and decoded keypoints.
        plt.figure()
        plt.imshow(cv2.cvtColor(crop_output, cv2.COLOR_BGRA2RGB))
        plt.imshow(heatmap.sum(axis=-1), alpha=0.7, interpolation="bilinear", cmap=plt.cm.get_cmap("viridis"))
        plt.show()

        # Check decode method.
        keypoints_decoded = encoder.decode(heatmap=heatmap, interpolate=True)
        is_keypoint_decoded = keypoints_decoded[:, 2] != 0
        is_keypoint_scaled = keypoints_scaled[:, 2] != 0
        print("Number of keypoints:", is_keypoint_decoded.sum(), is_keypoint_scaled.sum())
        print("Mean L2 error: ", np.mean(np.linalg.norm(keypoints_decoded[:, :2] - keypoints_scaled[:, :2], axis=-1)))
        print("Max L2 error: ", np.max(np.linalg.norm(keypoints_decoded[:, :2] - keypoints_scaled[:, :2], axis=-1)))

    # Close the windows.
    for window_name in window_names:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
