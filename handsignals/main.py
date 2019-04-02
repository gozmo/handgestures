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
import random

WIDTH = 640
HEIGHT = 400
def capture(frames_to_capture):
    frames_to_capture = int(frames_to_capture)
    frame_handling = FrameHandling()
    _ = frame_handling.collect_data(frames_to_capture)

def read_images(filepath, request):
    filename = os.path.basename(filepath)
    folder_path = os.path.dirname(filepath)

    return send_from_directory(folder_path, filename)

def train():
    trainer.train()

def resume_training():
    trainer.train(resume=True)

aided_batch_size = 10

def set_aided_annotation_batch_size(batch_size):
    global aided_batch_size
    aided_batch_size = batch_size

def aided_annotation():
    setup_model()

    dataset = ImageDataset(unlabel_data=True)

    random_indices = [random.randint(0, len(dataset)) for _ in range(30)]
    random_indices = list(set(random_indices))
    dataset = dataset.subdataset(random_indices)

    predictions = classify_dataset(dataset)

    value_extractor = lambda x: x[0]["score"]

    #the problem is here, label is set present and then set to None
    predictions_and_data = zip(predictions, dataset)
    predictions_and_data = sorted(predictions_and_data, key=value_extractor)

    aided = defaultdict(list)

    global aided_batch_size
    for (prediction, entry) in predictions_and_data[-aided_batch_size:]:
        filepath = entry["filepath"]
        filename = os.path.basename(filepath)
        label = prediction["label"]
        html_tuple=  (filename, prediction["distribution"])
        aided[label].append(html_tuple)

    all_labels = dataset.all_labels()

    return aided, all_labels

def annotate(annotation_dict):
    for filename, label in annotation_dict.items():
        print(f"Moving {filename} to {label}")
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
