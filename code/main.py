from models import get_model, get_basic_model


# TODO add anchors.
# TODO Make scale invariant (all size as inputs)
# TODO deeper/more blocks
# TODO metrics - mAP, compare to literature to see reasonable results.
# TODO NMS
# TODO input augmentation (e.g., train on several scales).
# TODO go multiclass.
# TODO transition into instance segmentation?
# TODO make it into YOLOv5-ish


def main():
    model1 = get_model(batch_normalization=False, dropout=False, activation="relu")
    model1.summary()
    model2 = get_basic_model()
    model2.summary()


if __name__ == "__main__":
    main()
