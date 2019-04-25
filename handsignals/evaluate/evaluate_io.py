import json
from collections import Counter
from handsignals.core import state
from handsignals.dataset import file_utils

def __write_content(filename, content):
    global_state = state.get_global_state()
    training_run_id = global_state.get_training_run_id()
    json_content= json.dumps(content)
    file_utils.write_evaluation_json(training_run_id, filename, json_content)


def write_loss(validation_loss, training_loss):
    __write_content("training_loss", training_loss)
    __write_content("validation_loss", validation_loss)

def write_prediction_results(prediction_results):
    json_results = [elem.to_json() for elem in prediction_results]

    __write_content("prediction_results", json_results)

def write_confusion_matrix(confusion_matrix):
    __write_content("confusion_matrix", confusion_matrix)

def write_dataset_stats(dataset):
    label_statistics = Counter(dataset.all_labels())
    dataset_statistics = {
        "dataset_size": len(dataset),
        "label_statistics": dict(label_statistics),
        }

    __write_content("dataset_stats", dataset_statistics)

def write_parameters(learning_rate, epochs, batch_size):
    parameters = {"learning_rate": learning_rate,
                  "epochs": epochs,
                  "batch_size": batch_size}
    __write_content("parameters", parameters)

def read_confusion_matrix(training_run_id):
    confusion_matrix = file_utils.read_evaluation_json(training_run_id, "confusion_matrix")
    return confusion_matrix
