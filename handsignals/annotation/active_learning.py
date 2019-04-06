from handsignals.networks.classify import setup_model
from handsignals.dataset.image_dataset import ImageDataset
import random
import os
from collections import defaultdict
from handsignals.networks.classify import classify_dataset

def generate_query(batch_size):
    setup_model()

    dataset = ImageDataset(unlabel_data=True)

    predictions = classify_dataset(dataset)

    value_extractor = lambda x: x[0].active_learning_score

    predictions_and_data = zip(predictions, dataset)
    predictions_and_data = sorted(predictions_and_data, key=value_extractor)

    aided = list()

    for (prediction, entry) in predictions_and_data[-batch_size:]:
        filepath = entry["filepath"]
        filename = os.path.basename(filepath)
        label = prediction.label
        html_tuple=  (filename, prediction.prediction_distribution)
        aided.append(html_tuple)

    return aided 
