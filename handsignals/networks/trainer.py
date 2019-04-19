from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import LabeledDataset
from handsignals.constants import Labels
from handsignals.constants import Event
import numpy as np
import torch
from handsignals.core.events import register_event

def train( learning_rate, epochs, batch_size,resume):
    register_event(Event.STARTED_TRAINING)

    dataset = LabeledDataset()

    conv_model = ConvNet(dataset.number_of_classes())
    if resume:
        conv_model.load()

    conv_model.train(dataset,
                     learning_rate=learning_rate,
                     epochs=epochs,
                     batch_size=batch_size)
    conv_model.save()
    del conv_model
    register_event(Event.TRAINING_DONE)
