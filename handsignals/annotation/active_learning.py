from handsignals.annotation.annotation_generator import generate
from handsignals.annotation.annotation_generator import AnnotationHolder

annotation_holder = AnnotationHolder(20)


def generate_query():
    global annotation_holder

    if len(annotation_holder) == 0:
        value_extractor = lambda x: x[0].active_learning_score

        query = generate(value_extractor)
        annotation_holder.set_annotations(query)

    batch = annotation_holder.get_batch()

    return batch
