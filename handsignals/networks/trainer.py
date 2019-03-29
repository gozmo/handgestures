from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.constants import Labels
import numpy as np
import torch

def train():
    dataset = ImageDataset()

    conv_model = ConvNet(dataset.number_of_classes())
    conv_model.train(dataset)
    conv_model.save()
