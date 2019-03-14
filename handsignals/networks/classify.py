from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.networks.simple_cnn import ConvNet
from handsignals.constants import Labels

import torch

def classify(dataset):

    labels = dataset.num_classes()
    conv_model = ConvNet(labels)
    conv_model.load

    predictions = []
    for idx in range(len(dataset)):
        d = dataset[idx]
        image = d["image"]

        prediction = conv_model.classify(image)

        _, prediction_idx = torch.max(prediction,1)
        prediction_idx = prediction_idx.data[0]
        prediction = Labels.int_to_label(prediction_idx)
        predictions.append(prediction)

    return predictions
