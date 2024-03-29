import copy
import random
import cv2
import os
import sys 
import json
import csv
import numpy as np
import shutil
from os import path
from handsignals.constants import Directories
from handsignals.constants import JsonAnnotation 

def _make_label_dir():
    try:
        os.makedirs(Directories.LABEL)
    except FileExistsError:
        pass


def move_image_to_label(source_path):
    _make_label_dir()
    destination = Directories.LABEL
    shutil.move(source_path, destination)
    print("Moved {} to {}".format(source_path, destination))


def move_file_to_label(filename):
    for dirpath, _, filenames in os.walk("dataset"):
        if filename in filenames:
            source_file = os.path.join(dirpath, filename)
            break
    move_image_to_label(source_file)


def get_images_paths():
    images = []
    for root, dirs, files in os.walk("dataset/unlabeled"):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith(".jpg"):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0 * w / h
            if aspect > 1.0 * WIDTH / HEIGHT:
                width = min(w, WIDTH)
                height = width / aspect
            else:
                height = min(h, HEIGHT)
                width = height * aspect
            images.append({"width": int(width), "height": int(height), "src": filename})
    return images


def read_image(filepath):
    image = cv2.imread(filepath)
    image_resized = cv2.resize(image, (320, 240))
    image_transposed = image_resized.transpose(2, 1, 0)
    normalized_image = image_transposed / 255
    return normalized_image


def get_labels():
    json_generator = filter(lambda x: "json" in x,
                            os.listdir(Directories.LABEL))
    all_labels = set()
    for json_file in json_generator:
        path = Directories.LABEL + json_file
        with open(path, "r") as f:
            json_content = json.load(f)
        labels = [annotation[JsonAnnotation.LABEL] for annotation in json_content]
        all_labels.update(labels)
    return list(all_labels)


def make_training_run_dir(training_run_id):
    path = f"training_runs/{training_run_id}"
    os.makedirs(path)
    return path

def basename(filename):
    if "jpg" in filename:
        return filename.replace(".jpg", "")
    elif "json" in filename:
        return filename.replace(".json", "")
    else:
        print("Unsupported filename:", filename)
        sys.exit()


def get_training_run_id():
    folders = os.listdir("evaluations")
    digit_folders = list(filter(str.isdigit, folders))
    if len(digit_folders) == 0:
        next_training_run_id = 0
    else:
        digits = map(int, digit_folders)
        max_digit = max(digits)
        next_training_run_id = max_digit + 1

    next_training_run_id = str(next_training_run_id)
    next_training_run_id = next_training_run_id.zfill(4)
    os.makedirs(f"evaluations/{next_training_run_id}")
    return next_training_run_id


def write_evaluation_json(training_run_id, filename, content):
    path = f"evaluations/{training_run_id}/{filename}.json"
    json_content = json.dumps(content)

    with open(path, "w") as f:
        f.write(json_content)


def read_evaluation_json(training_run_id, filename):
    path = f"evaluations/{training_run_id}/{filename}.json"

    with open(path, "r") as f:
        filecontent = f.read()

    json_content = json.loads(filecontent)
    json_content = json.loads(json_content)

    return json_content


def get_training_runs():
    training_runs = os.listdir("evaluations")
    return training_runs


def write_evaluation_csv(training_run_id, filename, content: dict):
    path = f"evaluations/{training_run_id}/{filename}.csv"
    with open(path, "a") as f:
        dict_writer = csv.DictWriter(f, content.keys())
        dict_writer.writeheader()
        dict_writer.writerow(content)


###
# Annotation functions
###

def __image_file_to_annotation_file(image_file):
    assert "jpg" in image_file

    return image_file.replace("jpg", "json")

def annotation_file_exists(image_file):
    csv_file = __image_file_to_annotation_file(image_file)
    path = Directories.LABEL + "/" + csv_file
    print("is file:", path)
    return os.path.isfile(path)

def read_annotations(image_file):
    json_file = __image_file_to_annotation_file(image_file)
    for dirpath, _, filenames in os.walk("dataset"):
        if json_file in filenames:
            source_file = os.path.join(dirpath, json_file)
            break

    with open(source_file, "r") as f:
        json_content = json.load(f)

    return json_content

def is_unlabeled(image_file):
    path = f"{Directories.UNLABEL}/{image_file}"
    return os.path.isfile(path)

def write_annotations(image_filename, annotations):
    print(f"writing annotations: {image_filename} {annotations}")
    json_filename = __image_file_to_annotation_file(image_filename)
    path = f"{Directories.LABEL}/{json_filename}"
    json_content = json.dumps(annotations)
    with open(path, "w") as f:
        f.write(json_content)
