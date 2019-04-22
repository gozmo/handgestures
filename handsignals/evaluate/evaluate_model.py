from handsignals.evaluate.evaluate_io import write_loss
from handsignals.evaluate.evaluate_io import write_prediction_results
from handsignals.evaluate.evaluate_io import write_confusion_matrix
from handsignals.evaluate.evaluate_io import write_prediction_distribution
from handsignals.dataset import file_utils

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
        confusion_matrix[result.true_label][result.predicted_label] += 1

    return confusion_matrix

def evaluate_model():

    # Holdout
    holdout_dataset = HoldoutDataset()
    holdout_results = evaluate_model_on_dataset(holdout_dataset)

    write_prediction_results(holdout_results)

    holdout_confusion_matrix = calculate_confusion_matrix(holdout_results)
    write_confusion_matrix(holdout_confusion_matrix)

    # Labeled dataset
    labeled_dataset = LabeledDataset()
    labeled_results = evaluate_model_on_dataset(labeled_dataset)

