import shutil
from os import path
import os

LABEL_PATH = "dataset/labeled/{}"
LABEL_IMAGE_PATH = "dataset/labeled/{}/{}"

def _get_destination(source_path, label):
    basename = path.basename(source_path)
    destination = LABEL_PATH.format(label, basename)
    return destination

def _make_label_dir(label):
    try:
        os.makedirs(LABEL_PATH.format(label))
    except FileExistsError:
        pass

def move_image_to_label(source_path, label):
    _make_label_dir(label)
    destination = _get_destination(source_path, label)
    shutil.move(source_path, destination)
    print("Moved {} to {}".format(source_path, destination))
