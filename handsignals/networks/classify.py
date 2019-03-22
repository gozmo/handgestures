from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.networks.simple_cnn import ConvNet
from handsignals.constants import Labels
from handsignals.dataset import file_utils

import torch

model = None



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
    prediction = model.classify(image)

    _, prediction_idx = torch.max(prediction,1)
    prediction_idx = prediction_idx.data[0]
    prediction = Labels.int_to_label(prediction_idx)
    return prediction

def classify_dataset(dataset):
    global model
    predictions = []
    for idx in range(len(dataset)):
        d = dataset[idx]
        image = d["image"]
        prediction = classify_image(image)

        predictions.append(prediction)

    return predictions
