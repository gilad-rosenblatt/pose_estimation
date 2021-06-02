from coco_anno_reader_fl import get_input_for_fl


# TODO install COCO API.
# TODO integrate/upgrade get_input from COCO method with COCO API.


class Parser:
    """Parse dataset annotations and navigate data folder."""

    # Dataset parent directory.
    PARENT_DIR = "../data"

    # Train and validation sets subdirectories.
    DATA_DIR_TRAIN = "train2017"
    DATA_DIR_VALIDATION = "val2017"

    # Annotations directory and filename for each set.
    ANNOTATIONS_DIR = "annotations"
    ANNOTATIONS_FILENAME_TRAIN = "instances_train2017.json"
    ANNOTATIONS_FILENAME_VALIDATION = "instances_val2017.json"

    def __init__(self, dataset):
        """
        :param str dataset: either "train" or "validation".
        """
        assert dataset == "train" or dataset == "validation", "Dataset input can be either 'train' or 'validation'."
        self._info = Parser._parse_annotations(dataset=dataset)
        self._data_dir = Parser._parse_data_directory(dataset=dataset)

    def get_path(self, filename):
        """
        :param str filename: name of an image file in the data folder.
        :return str: full path to the image file (to be used as input to image show method) assuming file exists.
        """
        return f"{self._data_dir}/{filename}"

    @property
    def info(self):
        """:return list: lists with filenames and their associated list of ground truth detection bounding boxes."""
        return self._info

    @staticmethod
    def _parse_annotations(dataset):
        if dataset == "train":
            info = get_input_for_fl(
                json_path=f"{Parser.PARENT_DIR}/{Parser.ANNOTATIONS_DIR}/{Parser.ANNOTATIONS_FILENAME_TRAIN}"
            )
        else:
            info = get_input_for_fl(
                json_path=f"{Parser.PARENT_DIR}/{Parser.ANNOTATIONS_DIR}/{Parser.ANNOTATIONS_FILENAME_VALIDATION}"
            )
        return info

    @staticmethod
    def _parse_data_directory(dataset):
        if dataset == "train":
            data_dir = f"{Parser.PARENT_DIR}/{Parser.DATA_DIR_TRAIN}"
        else:
            data_dir = f"{Parser.PARENT_DIR}/{Parser.DATA_DIR_VALIDATION}"
        return data_dir
