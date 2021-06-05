import itertools
import json
import os

import numpy as np

from metrics import ModelEvaluator


def tun_grid_scan():
    cls_thresholds = np.arange(0.5, 0.9, 0.05)
    nms_thresholds = np.arange(0.2, 0.8, 0.10)

    scorer = ModelEvaluator(model_filename="my_model_tim1622826272.136246_bsz64_epo21_ckp01")

    if not os.path.exists(ModelEvaluator.SCORES_DIR):
        os.makedirs(ModelEvaluator.SCORES_DIR)

    stats = []
    for cls_threshold, nms_threshold in itertools.product(cls_thresholds, nms_thresholds):
        print(f"Predicting ans scoring with cls_th = {cls_threshold:.2f}, nms_th = {nms_threshold:.2f}...")
        this_filename = scorer.predict_and_save(cls_threshold=0.8, nms_threshold=0.3)
        this_stats = ModelEvaluator.score(this_filename)
        stats.append(dict(
            cls_threshold=cls_threshold,
            nms_threshold=nms_threshold,
            stats=this_stats,
            map_50_to_95=this_stats[0],
            map_50=this_stats[1],
            map_75=this_stats[2],
        ))

    with open(os.path.join(ModelEvaluator.SCORES_DIR, "threshold_tuning.json"), "w") as out_file:
        json.dump(stats, out_file)


if __name__ == "__main__":
    tun_grid_scan()
