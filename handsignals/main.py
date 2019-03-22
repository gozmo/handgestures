from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from PIL import Image
from io import StringIO
from handsignals.networks import trainer
from handsignals.networks.classify import classify_dataset
from handsignals.networks.classify import classify_image
from handsignals.dataset.image_dataset import ImageDataset
import os
from collections import defaultdict
from handsignals.dataset import file_utils
from handsignals.networks.classify import setup_model
from handsignals.camera.frame_handling import FrameHandling

WIDTH = 640
HEIGHT = 400
def capture(frames_to_capture):
    frame_handling = FrameHandling()
    _ = frame_handling.collect_data(frames_to_capture)

def read_images(filepath, request):
    filename = os.path.basename(filepath)
    folder_path = os.path.dirname(filepath)

    return send_from_directory(folder_path, filename)

def train():
    trainer.train()

def aided_annotation():
    setup_model()

    dataset = ImageDataset(unlabel_data=True)
    predictions = classify_dataset(dataset)

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

live_view_active = False
def start_live_view():
    global live_view_active
    live_view_active = True

def live_view():
    global live_view_active
    if live_view_active:
        setup_model()
        frame_handling = FrameHandling()
        files = frame_handling.collect_data(1)
        image_path= files[0]
        image = file_utils.read_image(image_path)
        classification = classify_image(image)
        image_filename= os.path.basename(image_path)

    else:
        image_filename= ""
        classification = ""
    return image_filename, classification
