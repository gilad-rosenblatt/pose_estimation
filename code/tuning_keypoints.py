import itertools
import json
import os

import numpy as np

from metrics import KeypointsEvaluator


def get_tuning_full_file_name(model_filename):
    """
    Return full filename for the grid search json filename corresponding to the input model.

    :param str model_filename: filename of the model to run grid search on.
    :return str: full file name (with path) of the json file containing grid search results for the input model name.
    """
    return os.path.join(
        KeypointsEvaluator.SCORES_BASE_DIR,
        KeypointsEvaluator._get_model_type(),
        f"{KeypointsEvaluator._encode_string(model_filename)}_tuning.json"
    )


def run_grid_scan(model_filename):
    """
    Run grid search on classification threshold and non-max suppression thresholds on given model.

    :param str model_filename: filename of the model to run grid search on.
    """

    # Define threshold values to gris search over.
    thresholds = 1e-5 + np.arange(0.00, 0.80, 0.05)
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
    with open(get_tuning_full_file_name(model_filename=model_filename), "w") as out_file:
        json.dump(stats, out_file)


def get_best_hypers(model_filename, category="map_50"):
    """
    Get best threshold values for maximizing the given category of mAP score (reads grid search results from file).

    :param str model_filename: filename of the model to run grid search on.
    :param str category: score category to maximize (mAP@IoU=0.50, 0.75, or 0.50:0.95).
    :return tuple: maximum score and its corresponding hyperparameter values.
    """

    # Assert input category.
    assert category in ["map_50", "map_75", "map_50_to_95"]

    # Open model mAP stats from grid search.
    with open(get_tuning_full_file_name(model_filename=model_filename), "r") as in_file:
        stats = json.load(in_file)

    # Collect mAP @ IoU=0.50 scores for each hyperparameter combination and sort.
    scores = [(this_stats[category], this_stats["interpolate"], this_stats["threshold"]) for this_stats in stats]

    # Sort by descending scores and extract best hyperparameter combination.
    best_score, interpolate, threshold = sorted(scores, reverse=True)[0]

    # Return bets values.
    return best_score, interpolate, threshold


if __name__ == "__main__":
    filenames = dict(
        stage_1="my_model_tim1623386780.9279761_bsz64_epo15",  # 15ep@LR0.001&100ROP.  MSE all channels. No overfit.
        stage_2="my_model_tim1623401683.0648534_bsz64_epo40",  # 40ep@LR0.001&1000ROP. KP Y/N loss. Overfit.
        stage_3="my_model_tim1623423664.7099812_bsz64_epo7",  # 40ep@LR0.001&1000ROP. Disk mask loss. BAD.
        stage_4="my_model_junk_tim1623438554.4141183_bsz64_epo20",  # 20ep@LR0.001&1000ROP. KP Y/N loss. Bit overfit.
        stage_5="my_model_tim1623504195.0734112_bsz64_epo13"  # 13ep@LR0.001&1000ROP. KP Y/N loss. Best ckpt of stage 4.
    )
    # run_grid_scan(model_filename=filenames["stage_5"])
    for category in ["map_50", "map_75", "map_50_to_95"]:
        score, interpolate, threshold = get_best_hypers(model_filename=filenames["stage_5"], category=category)
        print(f"{category} score {score:.2f} for threshold = {threshold:.2f} and interpolate = {interpolate}.")
