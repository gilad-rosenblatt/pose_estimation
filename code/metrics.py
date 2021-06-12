import hashlib
import json
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from boxops import NMS
from datasets import DetectionsDataset, KeypointsDataset
from encoders import DetectionsEncoder, KeypointsEncoder
from parsers import DetectionsParser, KeypointsParser
from plotters import DetectionsPlotter, KeypointsPlotter


class ModelEvaluator(ABC):
    """Evaluator for calculating scores on models trained on COCO."""

    # Saved models and score files parent directories.
    MODELS_BASE_DIR = os.path.join("..", "models")
    SCORES_BASE_DIR = os.path.join("..", "scores")

    @classmethod
    def generate_results_file(cls, model_filename, **kwargs):
        """
        Collect and save the predicted detections of the model on the COCO validation set to a json file. Model
        post-processing hyperparameters can be provided as optional keyword arguments

        :param str model_filename: filename of the model to generate the results file for.
        """

        # Load dataset and and COCO API object.
        dataset = cls._dataset_factory()(batch_size=64, dataset="validation", generate_ids=True)
        coco_gt = COCO(annotation_file=cls._parser_factory().get_annotation_file(dataset="validation"))

        # Load model.
        model = tf.keras.models.load_model(
            filepath=cls._get_model_full_filename(model_filename=model_filename),
            compile=False
        )

        # Initialize encoder.
        encoder = cls._encoder_factory()(
            input_shape=cls._dataset_factory().INPUT_SHAPE,
            output_shape=cls._dataset_factory().OUTPUT_SHAPE
        )

        # Create the list of model predicted annotations.
        annotations = cls._generate_model_annotations(coco_gt, dataset, encoder, model, **kwargs)

        # Save collected annotations to a json results file whose name derives from model name and hyperparameters.
        full_filename = cls.get_score_full_filename(model_filename=model_filename, **kwargs)
        with open(full_filename, "w") as out_file:
            json.dump(annotations, out_file)

        # Return the filename.
        return full_filename

    @classmethod
    def score_results_file(cls, full_filename):
        """
        Score annotations saved to a results json file against the COCO validation ground truth.

        :param str full_filename: full filename for the annotation json file.
        """

        # Open results json file.
        with open(full_filename, "r") as in_file:
            detections = json.load(in_file)

        # Collect all image IDs used for prediction.
        ids = list(set([detection["image_id"] for detection in detections]))

        # Load the ground-truth and detection COCO annotations objects.
        coco_gt = COCO(annotation_file=cls._parser_factory().get_annotation_file(dataset="validation"))
        coco_dt = coco_gt.loadRes(resFile=full_filename)

        # Instantiate and configure and evaluation object.
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType=cls._get_iou_type())
        coco_eval.params.imgIds = ids
        coco_eval.params.catIds = [1]
        # coco_eval.params.areaRng = [[10000, 100000], [10000, 10001], [10001, 10002], [10002, 100000]]  # 0, 32^2, 96^2

        # Evaluate score and display to console.
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Return mAP stats.
        return coco_eval.stats

    @classmethod
    def show(cls, full_filename):
        """
        Observe annotations one by one (helps with debugging and visual evaluation).

        :param str full_filename: full filename for the annotation json file.
        """
        with open(full_filename, "r") as in_file:
            detections = json.load(in_file)
        cls._plotter_factory().show_annotations(detections)

    @classmethod
    def get_score_full_filename(cls, model_filename, **kwargs):
        full_filename = os.path.join(
            ModelEvaluator.SCORES_BASE_DIR,
            cls._get_model_type(),
            f"results_{ModelEvaluator._encode_string(model_filename)}"
        )
        for parameter, value in kwargs.items():
            full_filename += f"_{parameter[:3]}{float(value):1.2f}"  # _parX.XX
        return full_filename

    @classmethod
    def _get_model_full_filename(cls, model_filename):
        return os.path.join(ModelEvaluator.MODELS_BASE_DIR, cls._get_model_type(), model_filename)

    @staticmethod
    def _encode_string(string):
        """
        Return an 8-character encoding of a given (possibly long) string.

        :param str string: string to encode.
        :return str: 8-digit string encoding the input string.
        """
        return int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % 10 ** 8  # Only 8 digits.

    @classmethod
    @abstractmethod
    def _generate_model_annotations(cls, coco_gt, dataset, encoder, model, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def _parser_factory(cls):
        """Return a parser class constructor (callable)."""
        pass

    @classmethod
    @abstractmethod
    def _encoder_factory(cls):
        """Return a parser class constructor (callable)."""
        pass

    @classmethod
    @abstractmethod
    def _dataset_factory(cls):
        """Return a dataset class constructor (callable)."""
        pass

    @classmethod
    @abstractmethod
    def _plotter_factory(cls):
        """Return a plotter class constructor (callable)."""
        pass

    @classmethod
    @abstractmethod
    def _get_model_type(cls):
        """Return the model type to be scored (corresponds to sub-folder name)."""
        pass

    @classmethod
    @abstractmethod
    def _get_iou_type(cls):
        """Return the IOU type to be used for scoring as string (COCO: "bbox", "segm", "keypoints")."""
        pass


class DetectionsEvaluator(ModelEvaluator):
    """Evaluator for calculating mAP scores on object detection models trained on COCO."""

    @classmethod
    def _generate_model_annotations(cls, coco_gt, dataset, encoder, model, cls_threshold=0.8, nms_threshold=0.3):
        """
        Collect and save the predicted detections of the model on the COCO validation set to a annotations list.

        :param float cls_threshold: threshold used for 0/1 object classification when producing the annotations.
        :param float nms_threshold: threshold used for non-max suppression when producing the annotations.
        """

        # Initialize the list of annotations (detections).
        annotations = []

        # Predict for each batch and append resulting annotations.
        for batch_number, (x, _, image_ids) in enumerate(dataset):
            print(f"batch {batch_number} of {len(dataset) - 1}: num_images={x.shape[0]}.")

            # Make the prediction (threshold the class probability).
            y_prob = model.predict(x=x)
            y_pred = np.where(y_prob[..., 0:1] > cls_threshold, y_prob, 0)  # Classify by applying threshold score.

            # Decode and NMS-post-process boxes for each image and add to annotations list.
            for this_pred, image_id in zip(y_pred, image_ids):
                # Decode boxes from output and post-process using NMS.
                boxes_pred, scores = encoder.decode(this_y=this_pred)
                boxes_pred, scores = NMS.perform(boxes=boxes_pred, scores=scores, threshold=nms_threshold)

                # Resize boxes to fit the original image shape (as opposed to model output shape).
                this_image_dict = coco_gt.loadImgs(ids=[image_id])[0]
                this_image_shape = (this_image_dict["height"], this_image_dict["width"])
                boxes_pred = encoder.scale_boxes(boxes_pred, from_shape=dataset.INPUT_SHAPE, to_shape=this_image_shape)

                # Convert boxes to annotations and append to the annotations list.
                ones = np.ones(shape=scores.shape)
                result = np.concatenate((image_id * ones, boxes_pred, scores, ones), axis=1)
                annotations.extend(coco_gt.loadNumpyAnnotations(data=result))

        # Return model predicted annotations list.
        return annotations

    @classmethod
    @abstractmethod
    def _parser_factory(cls):
        """Return a parser class constructor (callable)."""
        return DetectionsParser

    @classmethod
    @abstractmethod
    def _encoder_factory(cls):
        """Return a parser class constructor (callable)."""
        return DetectionsEncoder

    @classmethod
    @abstractmethod
    def _dataset_factory(cls):
        """Return a dataset class constructor (callable)."""
        return DetectionsDataset

    @classmethod
    @abstractmethod
    def _plotter_factory(cls):
        """Return a plotter class constructor (callable)."""
        return DetectionsPlotter

    @classmethod
    def _get_model_type(cls):
        """Return the model type to be scored (corresponds to sub-folder name)."""
        return "detection"

    @classmethod
    def _get_iou_type(cls):
        """Return the IOU type to be used for scoring."""
        return "bbox"


class KeypointsEvaluator(ModelEvaluator):
    """Evaluator for calculating scores on keypoints detection models trained on COCO."""

    @classmethod
    def _generate_model_annotations(cls, coco_gt, dataset, encoder, model, interpolate=False):
        """
        Collect and save the predicted detections of the model on the COCO validation set to a annotations list.
        """

        # Initialize the list of results annotations.
        annotations = []

        # Predict for each batch and append resulting annotations.
        for batch_number, (x, _, annotation_ids) in enumerate(dataset):
            print(f"batch {batch_number} of {len(dataset) - 1}: num_images={x.shape[0]}.")

            # Make the prediction.
            y_pred = y_prob = model.predict(x=x)

            # Decode and post-process keypoints for each image and add to annotations list.
            for this_pred, this_annotation in zip(y_pred, coco_gt.loadAnns(ids=annotation_ids)):
                # Decode keypoints from output heatmap to input coordinates.
                keypoints_input = encoder.decode(heatmap=this_pred, interpolate=interpolate)

                # Extract the bounding box used to crop the original image in preprocessing.
                image_data = coco_gt.loadImgs(ids=[this_annotation["image_id"]])[0]
                box = KeypointsEncoder.expand_box(
                    box=np.array(this_annotation["bbox"], dtype=np.float32),  # Annotation detection box.
                    keypoints=np.array(this_annotation["keypoints"], dtype=np.float32).reshape(-1, 3),  # GT keypoints.
                    image_shape=(image_data["height"], image_data["width"]),  # Original input shape.
                    aspect_ratio=encoder.input_shape[1] / encoder.input_shape[0]  # Input width / height.
                )

                # Move keypoints back to coordinates of original image.
                keypoints_image = KeypointsDataset.upsize(box=box, keypoints=keypoints_input)

                # Save keypoint result in COCO format (<https://cocodataset.org/#format-results>).
                result = {
                    "image_id": this_annotation["image_id"],
                    "category_id": 1,  # Person.
                    "keypoints": np.round(keypoints_image).astype(int).ravel().tolist(),  # Use ints to save file size.
                    "score": 1
                }
                annotations.append(result)

        # Return model predicted annotations list.
        return annotations

    @classmethod
    @abstractmethod
    def _parser_factory(cls):
        """Return a parser class constructor (callable)."""
        return KeypointsParser

    @classmethod
    @abstractmethod
    def _encoder_factory(cls):
        """Return a parser class constructor (callable)."""
        return KeypointsEncoder

    @classmethod
    @abstractmethod
    def _dataset_factory(cls):
        """Return a dataset class constructor (callable)."""
        return KeypointsDataset

    @classmethod
    @abstractmethod
    def _plotter_factory(cls):
        """Return a plotter class constructor (callable)."""
        return KeypointsPlotter

    @classmethod
    def _get_model_type(cls):
        """Return the model type to be scored (corresponds to sub-folder name)."""
        return "keypoints"

    @classmethod
    def _get_iou_type(cls):
        """Return the IOU type to be used for scoring."""
        return "keypoints"


if __name__ == "__main__":
    evaluate_detections = False
    if evaluate_detections:
        # this_filename = DetectionsEvaluator.generate_results_file(
        #     model_filename="my_model_tim1622826272.136246_bsz64_epo21_ckp01",
        #     cls_threshold=0.8,
        #     nms_threshold=0.3
        # )
        this_filename = DetectionsEvaluator.get_score_full_filename(
            model_filename="my_model_tim1622826272.136246_bsz64_epo21_ckp01",
            cls_threshold=0.8,
            nms_threshold=0.3
        )
        stats = DetectionsEvaluator.score_results_file(this_filename)
        DetectionsEvaluator.show(this_filename)
    else:
        # this_filename = KeypointsEvaluator.generate_results_file(
        #     model_filename="my_model_tim1623386780.9279761_bsz64_epo15",
        #     threshold = 1e-5,
        #     interpolate=True
        # )
        this_filename = KeypointsEvaluator.get_score_full_filename(
            model_filename="my_model_tim1623386780.9279761_bsz64_epo15",
            threshold=1e-5,
            interpolate=True
        )
        stats = KeypointsEvaluator.score_results_file(this_filename)
        # KeypointsEvaluator.show(this_filename)  # TODO show_annotations not yet implemented.
