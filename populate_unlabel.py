import os
import random
from collections import defaultdict
from handsignals.constants import Directories
from handsignals.constants import Labels

def __read_labels():
    labels = os.listdir(Directories.LABEL)
    return labels

def __read_image_files(path):
    all_files = os.listdir(path)

    image_files = filter(lambda x: "jpg" in x, all_files)
    extend_path = lambda x: os.path.join(path, x)
    images_with_full_path = map(lambda x: extend_path(x), image_files)

    return list(images_with_full_path)

def __get_image_file_paths_and_label(directory, string_label):
    filepaths = __read_image_files(directory)
    file_labels = [string_label for x in range(len(filepaths))]
    return filepaths, file_labels

def get_files_to_move():
    labels = __read_labels()

    all_files = dict()

    for string_label in labels:
        filepath = f"{Directories.LABEL}/{string_label}"
        filepaths, filelabels = __get_image_file_paths_and_label(filepath, string_label)
        all_files[filelabels[0]] = filepaths

    picked_files = []
    for label in all_files:
        files = all_files[label]
        random.shuffle(files)
        picked_files.extend(files[0:2])

    return picked_files

files = get_files_to_move()
print(files)

for filepath in files:
    basename = os.path.basename(filepath)

    target = f"{Directories.UNLABEL}/{basename}"
    os.rename(filepath, target)
