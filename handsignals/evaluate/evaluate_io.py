import json
from handsignals.core import state
from handsignals.dataset import file_utils

def __write_content(filename, content):
    global_state = state.get_global_state()
    training_run_id = global_state.get_training_run_id()
    json_content= json.dumps(content)
    file_utils.write_evaluation_json(training_run_id, filename, json_content)

def write_loss(loss):
    __write_content("loss", loss)

def write_prediction_results(prediction_results):
    json_results = [elem.to_json() for elem in prediction_results]

    __write_content("prediction_results", json_results)

def write_confusion_matrix(confusion_matrix):
    __write_content("confusion_matrix", confusion_matrix)

