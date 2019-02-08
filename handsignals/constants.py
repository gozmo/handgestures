import tensorflow as tf

class Directories:
    UNLABEL = "dataset/unlabeled"
    LABEL = "dataset/labeled"
    TF_RECORDS = "dataset/tfrecords"

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
        return __labels[index] 


class TFRecord:
    IMAGE_FEATURE_DESCRIPTION = \
            { "label": tf.FixedLenFeature([], tf.int64),
              "text_label": tf.FixedLenFeature([], tf.string),
              "width": tf.FixedLenFeature([], tf.int64),
              "height": tf.FixedLenFeature([], tf.int64),
              "image": tf.FixedLenFeature([], tf.string)}


