class Directories:
    UNLABEL = "dataset/unlabeled/"
    LABEL = "dataset/labeled/"
    HOLDOUT = "dataset/holdout/"



class Labels:
    victory = "victory"
    metal = "metal"
    ok = "ok"
    none = "none"

    @staticmethod
    def get_labels():
        return [Labels.none, Labels.ok, Labels.metal, Labels.victory]

    @staticmethod
    def label_to_int(label):
        __labels = Labels.get_labels()
        int_label = __labels.index(label)
        return int_label

    @staticmethod
    def int_to_label(index):
        __labels = Labels.get_labels()
        label = __labels[index]
        return label


class Event:
    TRAINING_DONE = "training_done"

class TemplateFiles:
    ANNOTATE = "data/annotate.html"

class ImageAnnotation:
    X = "x"
    Y = "y"
    HEIGHT = "height"
    WIDTH = "width"
    LABEL = "label"
