from handsignals.networks.classify import setup_model
from handsignals.dataset.image_dataset import ImageDataset
import random
import os
from collections import defaultdict
from handsignals.networks.classify import classify_dataset
from handsignals.constants import Event
from handsignals.core import events

def generate(score_key):
    setup_model()

    dataset = ImageDataset(unlabel_data=True)

    predictions = classify_dataset(dataset)

    predictions_and_data = zip(predictions, dataset)
    predictions_and_data = sorted(predictions_and_data, key=score_key)

    return predictions_and_data 

class AnnotationHolder:
    def __init__(self, batch_size):
        self.__register_callback()
        self.__batch_size = batch_size
        self.__annotations = []

    def set_annotations(self, annotations):
        self.__annotations = annotations

    def get_batch(self):
        batch = self.__annotations[-self.__batch_size:]
        self.__annotations = self.__annotations[:-self.__batch_size]

        annotations = list()
        for (prediction, entry) in batch:
            filepath = entry["filepath"]
            filename = os.path.basename(filepath)
            label = prediction.label
            html_tuple=  (filename, prediction.prediction_distribution)
            annotations.append(html_tuple)
        return annotations

    def __register_callback(self):
        events.register_callback(Event.TRAINING_DONE, self.clear_cache)

    def clear_cache(self):
        self.__annotations = []

    def __len__(self):
        return len(self.__annotations)

