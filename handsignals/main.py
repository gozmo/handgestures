from flask import Flask, Response, request, abort, render_template_string, send_from_directory,render_template
from threading import Thread
import os

from handsignals.annotation import active_learning as al
from handsignals.annotation import aided_annotation as aa
from handsignals.camera.frame_handling import FrameHandling
from handsignals.constants import Labels
from handsignals.dataset import file_utils
from handsignals.networks import trainer
from handsignals.networks.classify import classify_image
from handsignals.evaluate import evaluate_io
from handsignals.evaluate.evaluate_model import evaluate_pipeline as evaluate_network
from handsignals.core.types import EvaluationResults
from handsignals.core.types import ModelParameters
from handsignals.core.state import state


WIDTH = 640
HEIGHT = 400
def capture(frames_to_capture):
    frames_to_capture = int(frames_to_capture)
    frame_handling = FrameHandling()
    _ = frame_handling.collect_data(frames_to_capture)

def read_images(filepath):
    filename = os.path.basename(filepath)
    folder_path = os.path.dirname(filepath)

    return send_from_directory(folder_path, filename)

def train(*args, **kwargs):
    model_parameters = ModelParameters(kwargs["learning_rate"],
                                       kwargs["epochs"],
                                       kwargs["batch_size"],
                                       kwargs["resume"])
    train = trainer.train
    training_thread = Thread(target=train, kwargs={"model_parameters":model_parameters})
    training_thread.start()


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
        model = state.get_application_model()
        frame_handling = FrameHandling()
        files = frame_handling.collect_data(1)
        image_path= files[0]
        image = file_utils.read_image(image_path)
        classification = classify_image(model, image)
        image_filename= os.path.basename(image_path)

    else:
        image_filename= ""
        classification = ""
    return image_filename, classification

def results(user_selected_training_run_id):
    training_runs = file_utils.get_training_runs()
    training_runs = sorted(training_runs, reverse=True)
    if user_selected_training_run_id is None:
        training_run_id = training_runs[0]
    else:
        training_run_id = user_selected_training_run_id

    results = EvaluationResults(training_run_id)

    results.add_label_order(file_utils.get_labels())

    results.add_parameters(evaluate_io.read_parameters(training_run_id))
    results.add_dataset_stats(evaluate_io.read_dataset_stats(training_run_id))

    evaluate_io.plot_loss_and_save_image(training_run_id)
    results.add_image("loss", "loss.jpg")

    results.add_matrix("holdout_confusion_matrix",
                       evaluate_io.read_confusion_matrix(training_run_id, "holdout"))
    evaluate_io.plot_prediction_distribution(training_run_id, "holdout")
    results.add_image("Holdout Prediction Distribution", "holdout_prediction_distribution")
    results.add_dictionary("f1_holdout",
                       evaluate_io.read_f1_score(training_run_id, "holdout"),
                       None,
                       None)



    results.add_matrix("labeled_confusion_matrix", evaluate_io.read_confusion_matrix(training_run_id, "labeled"))
    evaluate_io.plot_prediction_distribution(training_run_id, "labeled")
    results.add_image("Label Prediction Distribution", "labeled_prediction_distribution")
    results.add_dictionary("f1_labeled",
                       evaluate_io.read_f1_score(training_run_id, "labeled"),
                       None,
                       None)

    return training_runs, results

def evaluate_model():
    evaluate_network()

