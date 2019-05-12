from handsignals.networks.simple_cnn import ConvNet
from handsignals.networks.simple_cnn import ConvNet
from handsignals.constants import Labels
from handsignals.dataset import file_utils
from torch.utils.data import DataLoader
from handsignals.core.result import PredictionResult
from handsignals import device

import torch

model = None


def classification_results(prediction_distribution):
    label = predicted_label(prediction_distribution)
    score = prediction_score(prediction_distribution)

    distribution = prediction_distribution.tolist()
    return distribution, label, score


def setup_model():
    global model
    if model is None:
        labels = file_utils.get_labels()
        number_of_classes = len(labels)
        conv_model = ConvNet(number_of_classes)
        conv_model.load("torch.model")

        model = conv_model

def classify_image(image):
    global model

    prediction_distribution = model.classify(image)
    prediction_result = PredictionResult(prediction_distribution)

    return prediction_result


def classify_dataset(dataset):
    setup_model()
    global model
    predictions = []

    batch_size = 32

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)


    for batch in dataloader:
        a = torch.cuda.memory_allocated(device=device)
        images = batch["image"]
        labels = batch["label"]

        prediction_distributions =  model.classify_batch(images)

        for index in range(len(prediction_distributions)):
            prediction_distribution = prediction_distributions[index]
            label = labels[index]

            prediction_result = PredictionResult(prediction_distribution, label)

            predictions.append(prediction_result)

        del batch
        del prediction_distributions

    return predictions



