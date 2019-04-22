import copy
import random
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from handsignals.constants import Directories
from handsignals.constants import Labels
from handsignals.dataset import file_utils


class ImageDataset(Dataset):
    def __init__(self, folder):
        self.__folder = folder
        self.__available_labels = self.__read_labels()

        self.__read_files()

    def __read_files(self):
        if self.__folder == Directories.UNLABEL:
            self.__files = self.__get_unlabeled_image_file_paths()
            self.__labels = None
        elif self.__folder == Directories.LABEL:
            self.__files , self.__labels = self.__get_labeled_files()
        elif self.__folder == Directories.HOLDOUT:
            self.__files , self.__labels = self.__get_labeled_files()

    def __get_labeled_files(self):
        files = []
        all_labels = []
        for string_label in self.__available_labels:
            filepath = f"{Directories.LABEL}/{string_label}"
            filepaths, filelabels = self.__get_image_file_paths_and_label(filepath, string_label)
            files.extend(filepaths)
            all_labels.extend(filelabels)
        return files, all_labels

    def __get_unlabeled_image_file_paths(self):
        files = []
        filepaths, _ = self.__get_image_file_paths_and_label(Directories.UNLABEL, None)
        return filepaths

    def __get_image_file_paths_and_label(self, directory, string_label):
        filepaths = self.__read_image_files(directory)
        file_labels = [string_label for x in range(len(filepaths))]
        return filepaths, file_labels

    def __read_labels(self):
        labels = file_utils.get_labels()
        return labels

    def __read_image_files(self, path):
        all_files = os.listdir(path)

        image_files = filter(lambda x: "jpg" in x, all_files)
        extend_path = lambda x: os.path.join(path, x)
        images_with_full_path = map(lambda x: extend_path(x), image_files)

        return list(images_with_full_path)

    def __getitem__(self, idx):
        filepath = self.__files[idx]

        if self.__folder == Directories.UNLABEL:
            label_vector = [0]
        else:
            string_label = self.__labels[idx]
            label_int = Labels.label_to_int(string_label)
            label_vector = np.zeros(len(self.__available_labels))
            label_vector[label_int] = 1

        image = self.__read_image(filepath)

        return {"image": image,
                "label": label_vector,
                "filepath": filepath}

    def __read_image(self, filepath):
        image = file_utils.read_image(filepath)
        return image

    def __len__(self):
        return len(self.__files)

    def number_of_classes(self):
        return len(self.__available_labels)

    def all_labels(self):
        return self.__available_labels

    def subdataset(self, indices):
        sub_files = [self.__files[i] for i in indices]
        return ImageDataset(sub_files)

class UnlabeledDataset(ImageDataset):
    def __init__(self):
        super().__init__(Directories.UNLABEL)

class LabeledDataset(ImageDataset):
    def __init__(self):
        super().__init__(Directories.LABEL)

class HoldoutDataset(ImageDataset):
    def __init__(self):
        super().__init__(Directories.HOLDOUT)
