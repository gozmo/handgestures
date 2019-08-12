class Directories:
    UNLABEL = "dataset/unlabeled/"
    LABEL = "dataset/labeled/"
    HOLDOUT = "dataset/holdout/"



class Labels:
    ROCK = "rock"
    SCISSORS = "scissors"
    PAPER = "paper"
    HEAD = "head"


    @staticmethod
    def get_labels():
        return [Labels.ROCK, Labels.SCISSORS, Labels.PAPER, Labels.HEAD]

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

class TemplateFiles:
    ANNOTATE = "data/annotate.html"

class JsonAnnotation:
    X = "x"
    Y = "y"
    HEIGHT = "height"
    WIDTH = "width"
    LABEL = "label"
    LABELS = "labels"
    IMAGE = "image"
    IMAGE_FILENAME = "image_filename"
    JSON_FILENAME = "json_filename"
