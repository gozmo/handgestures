from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.constants import Labels
import numpy as np
import torch

def train( learning_rate, epochs, batch_size,resume):
    dataset = ImageDataset()

    conv_model = ConvNet(dataset.number_of_classes())
    if resume:
        conv_model.load()

    conv_model.train(dataset,
                     learning_rate=learning_rate,
                     epochs=epochs,
                     batch_size=batch_size)
    conv_model.save()
    del conv_model
