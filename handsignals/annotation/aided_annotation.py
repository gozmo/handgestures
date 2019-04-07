from handsignals.networks.classify import setup_model
from handsignals.dataset.image_dataset import ImageDataset
import random
import os
from collections import defaultdict
from handsignals.networks.classify import classify_dataset
from handsignals.annotation.annotation_generator import generate
from handsignals.annotation.annotation_generator import AnnotationHolder


annotation_holder = AnnotationHolder(20)

def annotate(aided_batch_size):
    global annotation_holder

    if len(annotation_holder) == 0:
        value_extractor = lambda x: x[0].score

        aided = generate(value_extractor)
        annotation_holder.set_annotations(aided)

    batch = annotation_holder.get_batch()

    return batch
