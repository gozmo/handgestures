from handsignals.networks.classify import setup_model
from handsignals.dataset.image_dataset import ImageDataset
import random
import os
from collections import defaultdict
from handsignals.networks.classify import classify_dataset

def annotate(aided_batch_size):
    setup_model()

    dataset = ImageDataset(unlabel_data=True)

    random_indices = [random.randint(0, len(dataset)) for _ in range(30)]
    random_indices = list(set(random_indices))
    dataset = dataset.subdataset(random_indices)

    predictions = classify_dataset(dataset)

    value_extractor = lambda x: x[0]["score"]

    #the problem is here, label is set present and then set to None
    predictions_and_data = zip(predictions, dataset)
    predictions_and_data = sorted(predictions_and_data, key=value_extractor)

    aided = defaultdict(list)

    for (prediction, entry) in predictions_and_data[-aided_batch_size:]:
        filepath = entry["filepath"]
        filename = os.path.basename(filepath)
        label = prediction["label"]
        html_tuple=  (filename, prediction["distribution"])
        aided[label].append(html_tuple)

    return aided 
