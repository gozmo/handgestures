from handsignals.evaluate.evaluate_io import write_prediction_results
from handsignals.evaluate.evaluate_io import write_confusion_matrix
from handsignals.evaluate.evaluate_io import write_f1_scores
from handsignals.dataset import file_utils
from handsignals.dataset.image_dataset import HoldoutDataset
from handsignals.dataset.image_dataset import LabeledDataset
from handsignals.dataset.image_dataset import ImageDataset


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


def calculate_confusion_matrix(results):
    confusion_matrix = dict()

    for label in file_utils.get_labels():
        inner_content = [(label, 0) for label in file_utils.get_labels()]
        inner_dict = dict(inner_content)
        confusion_matrix[label] = inner_dict

    for result in results:
        confusion_matrix[result.true_label][result.label] += 1

    return confusion_matrix

def evaluate_model_on_dataset(model, dataset, name_prefix):
    results = model.classify_dataset(dataset)

    write_prediction_results(results, name_prefix)

    confusion_matrix = calculate_confusion_matrix(results)
    write_confusion_matrix(confusion_matrix, name_prefix)

    f1_scores = __calculate_f1_scores(confusion_matrix)
    write_f1_scores(f1_scores, name_prefix)
    return results, confusion_matrix, f1_scores

def evaluate_pipeline(model):

    # Holdout
    holdout_dataset = HoldoutDataset()
    evaluate_model_on_dataset(model, holdout_dataset, "holdout")

    # Labeled dataset
    labeled_dataset = LabeledDataset()
    evaluate_model_on_dataset(model, labeled_dataset, "labeled")

