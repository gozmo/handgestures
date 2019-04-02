from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.constants import Labels
import numpy as np
import torch

def train(resume=False):
    dataset = ImageDataset()

    conv_model = ConvNet(dataset.number_of_classes())
    if resume:
        conv_model.load()
    conv_model.train(dataset)
    conv_model.save()
