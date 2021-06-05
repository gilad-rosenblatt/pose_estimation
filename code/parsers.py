import os
from collections import defaultdict

from pycocotools.coco import COCO


class Parser:
    """Parser for COCO dataset annotations and filenames."""

    # Dataset parent directory.
    PARENT_DIR = "../data"

    # Train and validation sets subdirectories.
    TRAIN_DIR = "train2017"
    VALIDATION_DIR = "val2017"

    def __init__(self, dataset):
        """
        :param str dataset: either "train" or "validation".
        """
        assert dataset == "train" or dataset == "validation", "Dataset input can be either 'train' or 'validation'."
        self._dataset = dataset
        self._info = Parser._get_annotations(filename=self._annotation_file)

    def get_path(self, filename):
        """
        :param str filename: name of an image file in the data folder.
        :return str: full path to the image file (to be used as input to image show method) assuming file exists.
        """
        return os.path.join(Parser.PARENT_DIR, self._data_type, filename)

    @property
    def info(self):
        """
        :return list: lists with filenames and their associated list of ground truth detection bounding boxes.
        """
        return self._info

    @property
    def _annotation_file(self):
        """
        :return str: annotations file name and full path.
        """
        return os.path.join(Parser.PARENT_DIR, "annotations", f"instances_{self._data_type}.json")

    @property
    def _data_type(self):
        """
        :return str: data sub-directory named according to "train" or "validation" data type.
        """
        return Parser.TRAIN_DIR if self._dataset == "train" else Parser.VALIDATION_DIR

    @staticmethod
    def _get_annotations(filename):
        """
        :param str filename: COCO dataset annotations filename (full path)".
        :return list: list of lists with filenames and their associated list of ground truth detection bounding boxes.
        """
        coco_gt = COCO(annotation_file=filename)
        boxes = defaultdict(list)
        for annotation in coco_gt.loadAnns(ids=coco_gt.getAnnIds(catIds=[1], areaRng=[10000, 100000])):  # Person + big.
            boxes[annotation["image_id"]].append(annotation["bbox"])
        return [(image["file_name"], boxes[image["id"]], image["id"]) for image in coco_gt.loadImgs(ids=boxes.keys())]
