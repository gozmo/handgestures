import json
from collections import Counter
from handsignals.core import state
from handsignals.dataset import file_utils
import matplotlib.pyplot as plt
from collections import defaultdict

def __write_content(filename, content):
    global_state = state.get_global_state()
    training_run_id = global_state.get_training_run_id()
    json_content= json.dumps(content)
    file_utils.write_evaluation_json(training_run_id, filename, json_content)


def write_loss(validation_loss, training_loss):
    __write_content("training_loss", training_loss)
    __write_content("validation_loss", validation_loss)

def write_prediction_results(prediction_results, prefix):
    json_results = [elem.to_json() for elem in prediction_results]

    __write_content(f"{prefix}-prediction_results", json_results)

def write_confusion_matrix(confusion_matrix, prefix):
    __write_content(f"{prefix}-confusion_matrix", confusion_matrix)

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

def write_f1_scores(f1_scores, prefix):
    __write_content(f"{prefix}-f1_scores", f1_scores)

def read_parameters(training_run_id):
    parameters = file_utils.read_evaluation_json(training_run_id, "parameters")
    return parameters

def read_confusion_matrix(training_run_id):
    confusion_matrix = file_utils.read_evaluation_json(training_run_id, "confusion_matrix")
    return confusion_matrix

def read_dataset_stats(training_run_id):
    dataset_stats = file_utils.read_evaluation_json(training_run_id, "dataset_stats")
    return dataset_stats

def read_f1_score(training_run_id):
    f1_scores= file_utils.read_evaluation_json(training_run_id, "f1_scores")
    return f1_scores

def plot_loss_and_save_image(training_run_id):
    training_loss= file_utils.read_evaluation_json(training_run_id, "training_loss")
    validation_loss = file_utils.read_evaluation_json(training_run_id, "validation_loss")

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(training_loss, label="training")
    ax.plot(validation_loss, label="validation")
    ax.legend()
    fig.savefig(f"evaluations/{training_run_id}/loss.jpg")
    return

def plot_prediction_distribution(training_run_id):
    results = file_utils.read_evaluation_json(training_run_id, "prediction_results")

    dists = defaultdict(list)
    for result_elem in results:
        for label, score in result_elem['prediction_distribution'].items():
            dists[label].append(score)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for label, dist in dists.items():
        ax.hist(dist, label=label, alpha=0.4)

    ax.legend()
    ax.set_xlim(0,1)
    fig.savefig(f"evaluations/{training_run_id}/prediction_distribution.jpg")
    return








