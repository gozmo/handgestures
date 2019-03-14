from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from PIL import Image
from io import StringIO
from handsignals.networks import trainer
from handsignals.networks.classify import classify
from handsignals.dataset.image_dataset import ImageDataset
import os
from collections import defaultdict
from handsignals.dataset import file_utils

WIDTH = 640
HEIGHT = 400
def capture(frames_to_capture):
    pass

def read_images(filepath, request):
    filename = os.path.basename(filepath)
    folder_path = os.path.dirname(filepath)

    return send_from_directory(folder_path, filename)

def train():
    trainer.train()

def aided_annotation():
    dataset = ImageDataset(unlabel_data=True)
    predictions = classify(dataset)

    aided = defaultdict(list)
    for entry, prediction in zip(dataset, predictions):
        filepath = entry["filepath"]
        filename = os.path.basename(filepath)
        aided[prediction].append(filename)

    all_labels = dataset.all_labels()

    return aided, all_labels

def annotate(annotation_dict):
    for filename, label in annotation_dict.items():
        file_utils.move_file_to_label(filename, label)

def active_learning():
    pass
