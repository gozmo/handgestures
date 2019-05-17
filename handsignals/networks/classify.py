from handsignals.dataset.image_dataset import ImageDataset
from handsignals.networks.base_network import BaseNetwork
from handsignals.core.result import PredictionResult
from handsignals import device

import torch


def classify_image(model:BaseNetwork, image):

    prediction_distribution = model.classify(image)
    prediction_result = PredictionResult(prediction_distribution)

    return prediction_result


def classify_dataset(model: BaseNetwork, dataset: ImageDataset):
    predictions = []

    dataloader = dataset.get_dataloader()

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



