from handsignals.networks.classify import setup_model
import random
import os
from collections import defaultdict
from handsignals.networks.classify import classify_dataset
from handsignals.annotation.annotation_generator import generate
from handsignals.annotation.annotation_generator import AnnotationHolder

annotation_holder = AnnotationHolder(20)

def generate_query(batch_size):
    global annotation_holder

    if len(annotation_holder) == 0:
        value_extractor = lambda x: x[0].active_learning_score

        query = generate(value_extractor)
        annotation_holder.set_annotations(query)

    batch = annotation_holder.get_batch()

    return batch

