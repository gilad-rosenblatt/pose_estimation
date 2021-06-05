import json
import hashlib
import os

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from boxops import NMS
from datasets import Dataset
from encoders import BoxEncoder
from plotters import Plotter
from parsers import Parser


class ModelEvaluator:
    """Evaluator for calculating mAP scores on object detection models trained on COCO."""

    # Saved models and score files parent directories.
    MODELS_DIR = "../models"  # TODO this should be stored somewhere else.
    SCORES_DIR = "../scores"

    def __init__(self, model_filename, dataset="validation"):
        """
        Initialize a model evaluator object and save the target model and dataset to score the model on.

        :param str model_filename: filename of the model.
        :param str dataset: either "train" or "validation".
        """
        self.model_filename = model_filename
        self.dataset = dataset

    def predict_and_save(self, cls_threshold=0.8, nms_threshold=0.3):
        """
        Collect and save the predicted detections of the model on the COCO validation set to a json file.

        :param float cls_threshold: threshold used for 0/1 object classification when producing the annotations.
        :param float nms_threshold: threshold used for non-max suppression when producing the annotations.
        """

        # Load dataset and and COCO API object.
        dataset = Dataset(batch_size=64, dataset=self.dataset, generate_image_ids=True)
        coco_gt = COCO(annotation_file=Parser.get_annotation_file(dataset=self.dataset))

        # Load model.
        model = tf.keras.models.load_model(os.path.join(ModelEvaluator.MODELS_DIR, self.model_filename), compile=False)

        # Initialize encoder (model output <--> boxes and scores).
        encoder = BoxEncoder(image_shape=dataset.IMAGE_SHAPE, cells_shape=dataset.CELLS_SHAPE)

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
                boxes_pred = encoder.scale(boxes_pred, from_shape=dataset.IMAGE_SHAPE, to_shape=this_image_shape)

                # Convert boxes to annotations and append to the annotations list.
                ones = np.ones(shape=scores.shape)
                result = np.concatenate((image_id * ones, boxes_pred, scores, ones), axis=1)
                annotations.extend(coco_gt.loadNumpyAnnotations(data=result))

        # Save collected annotations to a json results file whose name derives from model name and threshold values.
        filename = ModelEvaluator.get_filename(
            model_filename=self.model_filename,
            cls_threshold=cls_threshold,
            nms_threshold=nms_threshold
        )
        with open(os.path.join(ModelEvaluator.SCORES_DIR, filename), "w") as out_file:
            json.dump(annotations, out_file)

        # Return the filename.
        return filename

    @staticmethod
    def score(filename):
        """
        Score annotations saved to a results json file against the COCO validation ground truth.

        :param str filename: filename for the annotation json file.
        """

        # Open results json file.
        full_filename = os.path.join(ModelEvaluator.SCORES_DIR, filename)
        with open(full_filename, "r") as in_file:
            detections = json.load(in_file)

        # Collect all image IDs used for prediction.
        ids = list(set([detection["image_id"] for detection in detections]))

        # Load the ground-truth and detection COCO annotations objects.
        coco_gt = COCO(annotation_file=Parser.get_annotation_file(dataset="validation"))
        coco_dt = coco_gt.loadRes(resFile=full_filename)

        # Instantiate and configure and evaluation object.
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
        coco_eval.params.imgIds = ids
        coco_eval.params.catIds = [1]
        # coco_eval.params.areaRng = [[10000, 100000], [10000, 10001], [10001, 10002], [10002, 100000]]  # Default: 0, 32^2, 96^2

        # Evaluate score and display to console.
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Return mAP stats.
        return coco_eval.stats

    @staticmethod
    def show(filename):
        """
        Observe annotations one by one (helps with debugging and visual evaluation).

        :param str filename: filename for the annotation json file.
        """
        with open(os.path.join(ModelEvaluator.SCORES_DIR, filename), "r") as in_file:
            detections = json.load(in_file)
        Plotter.show_annotations(detections)

    @staticmethod
    def get_filename(model_filename, cls_threshold, nms_threshold):
        """
        Return a filename for the annotations json file for a model with the given name under given thresholds.

        :param str model_filename: filename of the model.
        :param float cls_threshold: threshold used for 0/1 object classification when producing the annotations.
        :param float nms_threshold: threshold used for non-max suppression when producing the annotations.
        :return str: filename for the annotation json file (including extension).
        """
        short_model_name = int(hashlib.sha1(model_filename.encode("utf-8")).hexdigest(), 16) % 10 ** 8  # Only 8 digits.
        return f"results_{short_model_name}_cls{cls_threshold:.2f}_nms{nms_threshold:.2f}"


if __name__ == "__main__":
    scorer = ModelEvaluator(model_filename="my_model_tim1622826272.136246_bsz64_epo21_ckp01")
    this_filename = scorer.predict_and_save(cls_threshold=0.8, nms_threshold=0.3)
    stats = ModelEvaluator.score(this_filename)
    ModelEvaluator.show(this_filename)
