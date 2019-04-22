from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import LabeledDataset
from handsignals.constants import Labels
from handsignals.constants import Event
import numpy as np
import torch
from handsignals.core.events import register_event
from handsignals.core import state
from handsignals.evaluate import evaluate_io
from handsignals.evaluate.evaluate_model import evaluate_model

def train(learning_rate, epochs, batch_size, resume):
    global_state = state.get_global_state()
    global_state.new_training_run()

    dataset = LabeledDataset()

    conv_model = ConvNet(dataset.number_of_classes())
    if resume:
        conv_model.load()

    loss = conv_model.train(dataset,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size)

    conv_model.save()

    evaluate_io.write_loss(loss)
    del conv_model

    evaluate_model()
