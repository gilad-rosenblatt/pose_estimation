import itertools
import json
import os

import numpy as np

from metrics import KeypointsEvaluator


def run_grid_scan(model_filename):
    """
    Run grid search on classification threshold and non-max suppression thresholds on given model.

    :param str model_filename: filename of the model to run grid search on.
    """

    # Define threshold values to gris search over.
    thresholds = np.arange(0.05, 0.80, 0.05)
    interpolates = [True, False]

    # Make sure a scores folder exists.
    if not os.path.exists(KeypointsEvaluator.SCORES_BASE_DIR):
        os.makedirs(KeypointsEvaluator.SCORES_BASE_DIR)

    # Scan over the gris of hyperparameters and collect score stats for each combination.
    stats = []
    for threshold, interpolate in itertools.product(thresholds, interpolates):
        print(f"Predicting ans scoring with threshold = {threshold:.2f}, interpolate = {interpolate}...")
        this_filename = KeypointsEvaluator.generate_results_file(
            model_filename=model_filename,
            interpolate=interpolate,
            threshold=threshold
        )
        this_stats = KeypointsEvaluator.score_results_file(this_filename)
        stats.append(dict(
            interpolate=int(interpolate),
            threshold=float(threshold),
            stats=[float(stat) for stat in this_stats],
            map_50_to_95=float(this_stats[0]),
            map_50=float(this_stats[1]),
            map_75=float(this_stats[2]),
        ))

    # Save all stats to json file.
    with open(os.path.join(KeypointsEvaluator.SCORES_BASE_DIR, "keypoints", "threshold_tuning.json"), "w") as out_file:
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
    with open(os.path.join(KeypointsEvaluator.SCORES_BASE_DIR, "keypoints", "threshold_tuning.json"), "r") as in_file:
        stats = json.load(in_file)

    # Collect mAP @ IoU=0.50 scores for each hyperparameter combination and sort.
    scores = [(this_stats[category], this_stats["interpolate"], this_stats["threshold"]) for this_stats in stats]

    # Sort by descending scores and extract best hyperparameter combination.
    best_score, interpolate, threshold = sorted(scores, reverse=True)[0]

    # Return bets values.
    return best_score, interpolate, threshold


if __name__ == "__main__":
    run_grid_scan(model_filename="my_model_tim1623504195.0734112_bsz64_epo13")
    for category in ["map_50", "map_75", "map_50_to_95"]:
        score, interpolate, threshold = get_best_hypers(category=category)
        print(f"{category} score {score:.2f} for threshold = {threshold:.2f} and interpolate = {interpolate}.")
