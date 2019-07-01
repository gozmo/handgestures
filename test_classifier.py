from handsignals.networks.torch_alexnet import AlexNet
from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.constants import Labels
import random
import numpy as np
import torch

dataset = ImageDataset()
indices = [random.randint(0, len(dataset)) for x in range(20)]


# a = AlexNet(dataset.num_classes())
a = ConvNet(dataset.num_classes())
a.train(dataset)

for x in indices:
    d = dataset[x]
    image = d["image"]
    label = d["label"]
    prediction = a.classify(image)
    _, prediction_idx = torch.max(prediction, 1)
    prediction_idx = prediction_idx.data[0]
    label_idx = np.argmax(label)

    # print(label, prediction.data[0])
    print(
        "Label: ",
        Labels.int_to_label(label_idx),
        "Predicted:",
        Labels.int_to_label(prediction_idx),
    )
