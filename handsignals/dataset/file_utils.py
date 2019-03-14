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

def move_file_to_label(filename, label):
    for dirpath, _, filenames in os.walk("dataset"):
        if filename in filenames:
            source_file =  os.path.join(dirpath, filename)
    move_image_to_label(source_file, label)

def get_images_paths():
    images = []
    for root, dirs, files in os.walk('dataset/unlabeled'):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.jpg'):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0*w/h
            if aspect > 1.0*WIDTH/HEIGHT:
                width = min(w, WIDTH)
                height = width/aspect
            else:
                height = min(h, HEIGHT)
                width = height*aspect
            images.append({
                'width': int(width),
                'height': int(height),
                'src': filename
            })
    return images
