from PIL import Image
from collections import defaultdict
from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from io import StringIO
from threading import Thread
import os
import random

from handsignals.annotation import active_learning as al
from handsignals.annotation import aided_annotation as aa
from handsignals.camera.frame_handling import FrameHandling
from handsignals.constants import Labels
from handsignals.dataset import file_utils
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.networks import trainer
from handsignals.networks.classify import classify_dataset
from handsignals.networks.classify import classify_image
from handsignals.networks.classify import setup_model

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

def train(*args, **kwargs):
    train = trainer.train
    threading = Thread(target=train, kwargs=kwargs)
    threading.start()


aided_batch_size = 50

def set_aided_annotation_batch_size(batch_size):
    global aided_batch_size
    aided_batch_size = batch_size

def aided_annotation():
    annotation_help = aa.annotate(aided_batch_size)
    all_labels = Labels.get_labels()
    return annotation_help, all_labels

def annotate(annotation_dict):
    for filename, label in annotation_dict.items():
        print(f"Moving {filename} to {label}")
        file_utils.move_file_to_label(filename, label)

active_learning_batch_size = 30 
def set_aided_annotation_batch_size(batch_size):
    global active_learning_batch_size 
    aided_batch_size = batch_size

def active_learning():
    global active_learning_batch_size 
    annotation_help = al.generate_query(active_learning_batch_size)
    all_labels = Labels.get_labels()
    return annotation_help, all_labels

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
