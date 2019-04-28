from handsignals.evaluate.evaluate_io import write_loss
from handsignals.evaluate.evaluate_io import write_prediction_results
from handsignals.evaluate.evaluate_io import write_confusion_matrix
from handsignals.evaluate.evaluate_io import write_f1_scores
from handsignals.dataset import file_utils
from handsignals.dataset.image_dataset import HoldoutDataset
from handsignals.dataset.image_dataset import LabeledDataset
from handsignals.networks.classify import classify_dataset

def __calculate_precision_and_recall(working_label, labels, confusion_matrix):
    recall_sum = 0
    precision_sum = 0
    for label in labels:

        precision_sum += confusion_matrix[label][working_label]
        recall_sum += confusion_matrix[working_label][label]

    tp = confusion_matrix[working_label][working_label]

    try:
        recall = tp / recall_sum
    except ZeroDivisionError:
        recall = 0

    try:
        precision = tp / precision_sum
    except ZeroDivisionError:
        precision = 0

    return recall, precision

def __calculate_f1_scores(confusion_matrix):
    labels = file_utils.get_labels()
    scores = dict()

    f1_sum = 0

    for label in labels:
        recall, precision = __calculate_precision_and_recall(label, labels, confusion_matrix)

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        label_scores = {"f1": f1,
                        "recall": recall,
                        "precision": precision}
        scores[label] = label_scores

        f1_sum += f1

    f1_score = f1_sum / len(labels)

    evaluation_score = {"f1": f1_score,
                        "label_f1": scores}

    return evaluation_score

def evaluate_model_on_dataset(dataset):
    predictions = classify_dataset(dataset)
    return predictions

def calculate_confusion_matrix(results):
    confusion_matrix = dict()

    for label in file_utils.get_labels():
        inner_content = [(label, 0) for label in file_utils.get_labels()]
        inner_dict = dict(inner_content)
        confusion_matrix[label] = inner_dict

    for result in results:
        confusion_matrix[result.true_label][result.label] += 1

    return confusion_matrix

def evaluate_model():

    # Holdout
    holdout_dataset = HoldoutDataset()
    holdout_results = evaluate_model_on_dataset(holdout_dataset)

    write_prediction_results(holdout_results)

    holdout_confusion_matrix = calculate_confusion_matrix(holdout_results)
    write_confusion_matrix(holdout_confusion_matrix)

    f1_scores = __calculate_f1_scores(holdout_confusion_matrix)
    write_f1_scores(f1_scores)

    # Labeled dataset
    labeled_dataset = LabeledDataset()
    labeled_results = evaluate_model_on_dataset(labeled_dataset)

