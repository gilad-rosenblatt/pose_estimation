import os
from abc import ABC, abstractmethod
from collections import defaultdict

from pycocotools.coco import COCO


class Parser(ABC):
    """Parser for COCO dataset annotations and filenames."""

    # Dataset parent directory.
    PARENT_DIR = "../data"

    # Train and validation sets subdirectories.
    TRAIN_DIR = "train2017"
    VALIDATION_DIR = "val2017"

    def __init__(self, dataset):
        """
        Initialize a parser object for the given dataset and parse its annotations.

        :param str dataset: either "train" or "validation".
        """
        assert dataset == "train" or dataset == "validation", "Dataset input can be either 'train' or 'validation'."
        self._dataset = dataset
        self._info = self._get_annotations(filename=self.get_annotation_file(dataset=dataset))

    @property
    def info(self):
        """:return list: list of sequences each with an image filename and its associated annotations of choice."""
        return self._info

    def get_path(self, filename):
        """
        Return the full path and filename of the image corresponding to the input filename.

        :param str filename: name of an image file in the data folder.
        :return str: full path to the image file (to be used as input to image show method) assuming file exists.
        """
        return os.path.join(Parser.PARENT_DIR, Parser.get_data_dir(dataset=self._dataset), filename)

    @classmethod
    def get_annotation_file(cls, dataset):
        """
        :param str dataset: dataset to return the sub-directory name for ("train" or "validation").
        :return str: annotations file name and full path.
        """
        assert dataset == "train" or dataset == "validation", "Dataset input can be either 'train' or 'validation'."
        return os.path.join(
            Parser.PARENT_DIR, "annotations", f"{cls.annotation_type()}_{Parser.get_data_dir(dataset=dataset)}.json"
        )

    @staticmethod
    def get_data_dir(dataset):
        """
        :param str dataset: dataset to return the sub-directory name for ("train" or "validation").
        :return str: sub-directory name containing the dataset images.
        """
        assert dataset == "train" or dataset == "validation", "Dataset input can be either 'train' or 'validation'."
        return Parser.TRAIN_DIR if dataset == "train" else Parser.VALIDATION_DIR

    @staticmethod
    @abstractmethod
    def annotation_type():
        """The annotation type prefix string to use when constructing the annotation filename in get_annotation_file."""
        pass

    @staticmethod
    @abstractmethod
    def _get_annotations(filename):
        """Generate list of filenames and corresponding annotations from the COCO annotation file (to populate info)."""
        pass


class DetectionsParser(Parser):
    """Parser for COCO dataset bounding box annotations and filenames (for detection tasks)."""

    @staticmethod
    def annotation_type():
        """:return str: the instances annotation type prefix in the annotation file name."""
        return "instances"

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


class KeypointsParser(Parser):

    @staticmethod
    def annotation_type():
        """:return str: the keypoints annotation type prefix in the annotation file name."""
        return "person_keypoints"

    @staticmethod
    def _get_annotations(filename):
        """
        :param str filename: COCO dataset annotations filename (full path)".
        :return list: list of tuples with a filename and an associated ground truth bounding box and its keypoints.
        """
        coco_gt = COCO(annotation_file=filename)
        return [
            (coco_gt.loadImgs(ids=annotation["image_id"])[0]["file_name"], annotation["bbox"], annotation["keypoints"])
            for annotation in coco_gt.loadAnns(ids=coco_gt.getAnnIds(catIds=[1]))  # All person annotations.
            if not annotation["iscrowd"] and annotation["num_keypoints"] > 0 and annotation["area"] > 0  # Drop empties.
        ]


if __name__ == "__main__":
    """Fast test to see contents in visual debugger."""
    parser = KeypointsParser(dataset="validation")
    print("Loaded successfully.")
