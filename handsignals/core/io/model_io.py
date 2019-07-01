from handsignals.dataset import file_utils
from handsignals.networks.simple_cnn import ConvNet


def __load_model(path: str):
    labels = file_utils.get_labels()
    number_of_classes = len(labels)
    conv_model = ConvNet(number_of_classes)
    conv_model.load(path)

    return conv_model


def load_model(training_run_id):
    model = __load_model(f"evaluations/{training_run_id}/torch.model")

    return model


def load_application_model():
    model = __load_model("torch.model")

    return model
