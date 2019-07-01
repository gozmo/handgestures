import os
import tensorflow as tf
import numpy as np
from PIL import Image
from handsignals.constants import Directories
from handsignals.constants import Labels
from handsignals.constants import TFRecord


class CreateTFRecords:
    def create_tf_records(self):
        files_and_labels = self.__get_image_file_paths_and_label()

        tf_record_name = os.path.join(Directories.TF_RECORDS, "test.tfrecords")
        with tf.python_io.TFRecordWriter(tf_record_name) as writer:
            print(f"Writing to record: {tf_record_name}")

            for (filepath, label) in files_and_labels:
                image = Image.open(filepath)
                image = image.resize((224, 224))
                image_raw = np.array(image).tostring()
                int_label = Labels.label_to_int(label)

                features = {
                    "label": self._int64_feature(int_label),
                    "text_label": self._bytes_feature(label.encode()),
                    "width": self._int64_feature(224),
                    "height": self._int64_feature(224),
                    "image": self._bytes_feature(image_raw),
                }
                tf_features = tf.train.Features(feature=features)
                example = tf.train.Example(features=tf_features)
                writer.write(example.SerializeToString())

    def __get_image_file_paths_and_label(self):
        labels = self.__read_labels()

        files_and_labels = []
        for label in labels:
            labeled_images = self.__read_image_files(label)
            files_and_labels.extend(labeled_images)
        return files_and_labels

    def __read_labels(self):
        labels = os.listdir(Directories.LABEL)
        return labels

    def __read_image_files(self, label):
        path = os.path.join(Directories.LABEL, label)
        all_files = os.listdir(path)
        image_files = filter(lambda x: "jpg" in x, all_files)
        extend_path = lambda x: os.path.join(path, x)
        images_and_label = map(lambda x: (extend_path(x), label), image_files)

        return images_and_label

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_function(example_proto):
    features = {
        "label": tf.FixedLenFeature((), tf.int64, default_value=0),
        "text_label": tf.FixedLenFeature((), tf.string, default_value=""),
        "width": tf.FixedLenFeature((), tf.int64, default_value=224),
        "height": tf.FixedLenFeature((), tf.int64, default_value=224),
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    img_decoded = tf.image.decode_image(parsed_features["image"], channels=1)
    img_decoded = tf.image.convert_image_dtype(img_decoded, dtype=tf.float32)
    img_decoded.set_shape([224, 224, 1])  # This line was missing
    label = tf.cast(parsed_features["label"], tf.int32)
    return img_decoded, label


def read_tf_record(placeholder_filenames):
    dataset = tf.data.TFRecordDataset(placeholder_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(32)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.take(1)

    iterator = dataset.make_initializable_iterator()
    return iterator


def read_classification_images():
    pass


if __name__ == "__main__":
    create_tf_records = CreateTFRecords()
    create_tf_records.create_tf_records()
    read_tf_record()
