import copy
import random
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from handsignals.constants import Directories
from handsignals.constants import Labels

class ImageDataset:
    def __init__(self):
        self.__available_labels = self.__read_labels()
        self.__files, self.__labels = self.__get_image_file_paths_and_label()

    def __get_image_file_paths_and_label(self):

        files = []
        all_labels = []
        for string_label in self.__available_labels:
            filepaths = self.__read_image_files(string_label)
            file_labels = [string_label for x in range(len(filepaths))]
            files.extend(filepaths)
            all_labels.extend(file_labels)

        return files, all_labels

    def __read_labels(self):
        labels = os.listdir(Directories.LABEL)
        return labels

    def __read_image_files(self, label):
        path = os.path.join(Directories.LABEL, label)
        all_files = os.listdir(path)

        image_files = filter(lambda x: "jpg" in x, all_files)
        extend_path = lambda x: os.path.join(path, x)
        images_with_full_path = map(lambda x: extend_path(x), image_files)

        return list(images_with_full_path)

    def __getitem__(self, idx):
        filepath = self.__files[idx]
        string_label = self.__labels[idx]
        label_int = Labels.label_to_int(string_label)
        label_vector = np.zeros(len(self.__available_labels))
        label_vector[label_int] = 1

        image = self.__read_image(filepath)

        return {"image": image, "label": label_vector}

    def __read_image(self, filepath):
        image = cv2.imread(filepath)
        image_resized = cv2.resize(image, (11,11))
        image_transposed = image_resized.transpose(2,1,0)
        normalized_image = image_transposed/ 255
        return normalized_image

    def __len__(self):
        return len(self.__files)

    def num_classes(self):
        return len(self.__available_labels)
