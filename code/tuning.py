import itertools
import json
import os

import numpy as np

from metrics import DetectionsEvaluator


def run_grid_scan(model_filename):
    """
    Run grid search on classification threshold and non-max suppression thresholds on given model.

    :param str model_filename: filename of the model to run grid search on.
    """

    # Define threshold values to gris search over.
    cls_thresholds = np.arange(0.5, 0.9, 0.05)
    nms_thresholds = np.arange(0.2, 0.8, 0.10)

    # Initialize a model evaluator (mAP over COCO validation).
    scorer = DetectionsEvaluator(model_filename=model_filename)

    # Make sure a scores folder exists.
    if not os.path.exists(DetectionsEvaluator.SCORES_DIR):
        os.makedirs(DetectionsEvaluator.SCORES_DIR)

    # Scan over the gris of hyperparameters and collect score stats for each combination.
    stats = []
    for cls_threshold, nms_threshold in itertools.product(cls_thresholds, nms_thresholds):
        print(f"Predicting ans scoring with cls_th = {cls_threshold:.2f}, nms_th = {nms_threshold:.2f}...")
        this_filename = scorer.generate_results_file(cls_threshold=cls_threshold, nms_threshold=nms_threshold)
        this_stats = DetectionsEvaluator.score_results_file(this_filename)
        stats.append(dict(
            cls_threshold=float(cls_threshold),
            nms_threshold=float(nms_threshold),
            stats=[float(stat) for stat in this_stats],
            map_50_to_95=float(this_stats[0]),
            map_50=float(this_stats[1]),
            map_75=float(this_stats[2]),
        ))

    # Save all stats to json file.
    with open(os.path.join(DetectionsEvaluator.SCORES_DIR, "threshold_tuning.json"), "w") as out_file:
        json.dump(stats, out_file)


def get_best_hypers(category="map_50"):
    """
    Get best threshold values for maximizing the given category of mAP score (reads grid search results from file).

    :param str category: score category to maximize (mAP@IoU=0.50, 0.75, or 0.50:0.95).
    :return tuple: maximum score and its the classification and NMS threshold values.
    """

    # Assert input category.
    assert category in ["map_50", "map_75", "map_50_to_95"]

    # Open model mAP stats from grid search.
    with open(os.path.join(DetectionsEvaluator.SCORES_DIR, "threshold_tuning.json"), "r") as in_file:
        stats = json.load(in_file)

    # Collect mAP @ IoU=0.50 scores for each hyperparameter combination and sort.
    scores = [(this_stats[category], this_stats["cls_threshold"], this_stats["nms_threshold"]) for this_stats in stats]

    # Sort by descending scores and extract best hyperparameter combination.
    best_score, cls_threshold, nms_threshold = sorted(scores, reverse=True)[0]

    # Return bets values.
    return best_score, cls_threshold, nms_threshold


if __name__ == "__main__":
    # run_grid_scan(model_filename="my_model_tim1622826272.136246_bsz64_epo21_ckp01")
    for category in ["map_50", "map_75", "map_50_to_95"]:
        score, cls_th, nms_th = get_best_hypers(category=category)
        print(f"{category} score {score:.2f} for cls_th = {cls_th:.2f} and nms_th = {nms_th:.2f}.")
