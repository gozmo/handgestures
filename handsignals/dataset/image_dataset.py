import copy
import json
import random
import cv2
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from handsignals.constants import Directories
from handsignals.constants import Labels
from handsignals.constants import JsonAnnotation
from handsignals.dataset import file_utils


class ImageDataset(Dataset):
    def __init__(self, folder):
        self.__folder = folder
        self.__available_labels = self.__read_labels()

        self.__read_files()

    def __read_files(self):
        if self.__folder == Directories.LABEL:
            self.__files = self.__get_labeled_files()

        # elif self.__folder == Directories.UNLABEL:
            # self.__files = self.__get_unlabeled_image_file_paths()
            # self.__labels = None
        # elif self.__folder == Directories.HOLDOUT:
            # self.__files, self.__labels = self.__get_labeled_files()

    def __get_labeled_files(self):
        files = list() 
        json_generator = filter(lambda x: "json" in x,
                                os.listdir(Directories.LABEL))
        for json_file in json_generator:
            path = Directories.LABEL + json_file
            with open(path, "r") as f:
                labels = json.load(f)

            basename = file_utils.basename(json_file)

            dataset_entry = DatasetElem(basename + ".jpg",
                                        json_file,
                                        labels)
            files.append(dataset_entry)
        return files

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
        dataset_elem = self.__files[idx]


        if self.__folder == Directories.UNLABEL:
            coco_annotations= []
        else:
            labels = dataset_elem.labels
            coco_annotation = self.__convert_annotations(labels)

        image = self.__read_image(dataset_elem.image_filename)
        dataset_image = DatasetImage(image,
                                     coco_annotations)
        return dataset_image

    def __read_image(self, filepath):
        image = file_utils.read_image(filepath)
        return image

    def __convert_annotations(self, labels):
        coco_labels = []
        for label in labels:
            new_label = {
                JsonAnnotation.X : float(label[JsonAnnotation.X]) + float(label[JsonAnnotation.WIDTH]) / 2,
                JsonAnnotation.Y : float(label[JsonAnnotation.Y]) + float(label[JsonAnnotation.HEIGHT]) / 2,
                JsonAnnotation.WIDTH : float(label[JsonAnnotation.WIDTH]) / 2,
                JsonAnnotation.HEIGHT: float(label[JsonAnnotation.HEIGHT]) / 2}
            coco_labels.append(new_label)

        return coco_labels

    def __len__(self):
        return len(self.__files)

    def number_of_classes(self):
        return len(self.__available_labels)

    def all_labels(self):
        return self.__labels

    def subdataset(self, indices):
        sub_files = [self.__files[i] for i in indices]
        return ImageDataset(sub_files)

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class UnlabeledDataset(ImageDataset):
    def __init__(self):
        super().__init__(Directories.UNLABEL)


class LabeledDataset(ImageDataset):
    def __init__(self):
        super().__init__(Directories.LABEL)


class HoldoutDataset(ImageDataset):
    def __init__(self):
        super().__init__(Directories.HOLDOUT)

class DatasetElem:
    def __init__(self, image_filename, json_filename, labels):
        self.image_filename = image_filename
        self.json_filename = json_filename,
        self.labels = labels
