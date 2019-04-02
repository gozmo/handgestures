from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.networks.simple_cnn import ConvNet
from handsignals.constants import Labels
from handsignals.dataset import file_utils

import torch

model = None

def prediction_score(prediction_distribution):
    score, prediction_idx = torch.max(prediction_distribution,1)
    return score.item()

def predicted_label(prediction_distribution):
    _, prediction_idx = torch.max(prediction_distribution,1)
    prediction_idx = prediction_idx.data[0]
    label = Labels.int_to_label(prediction_idx)
    return label


def setup_model():
    global model
    if model is None:
        labels = file_utils.get_labels()
        number_of_classes = len(labels)
        conv_model = ConvNet(number_of_classes)
        conv_model.load

        model = conv_model

def classify_image(image):
    global model

    prediction_distribution = model.classify(image)

    label = predicted_label(prediction_distribution)
    score = prediction_score(prediction_distribution)

    distribution = prediction_distribution.tolist()[0]
    return distribution, label, score

def classify_dataset(dataset):
    global model
    predictions = []
    for idx in range(len(dataset)):
        d = dataset[idx]
        image = d["image"]

        prediction_distribution, label, score = classify_image(image)

        distribution = dict(zip(Labels.get_labels(), prediction_distribution))
        entry = {"distribution": distribution,
                 "score": score,
                 "label": label}

        predictions.append(entry)

    return predictions



