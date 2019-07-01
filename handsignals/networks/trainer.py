from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import LabeledDataset
from handsignals.dataset.image_dataset import HoldoutDataset
from handsignals.core import state
from handsignals.evaluate import evaluate_io
from handsignals.evaluate.evaluate_model import evaluate_pipeline


def train(model_parameters):
    global_state = state.get_global_state()
    global_state.new_training_run()

    training_dataset = LabeledDataset()
    holdout_dataset = HoldoutDataset()

    evaluate_io.write_parameters(model_parameters)
    evaluate_io.write_dataset_stats(training_dataset)

    conv_model = ConvNet(training_dataset.number_of_classes())
    if model_parameters.resume:
        conv_model.load()

    conv_model.train(training_dataset, holdout_dataset, model_parameters)

    path = global_state.training_run_folder()
    conv_model.save(path + "/torch.model")
    conv_model.save("./torch.model")

    # evaluate_io.write_loss(validation_loss, training_loss)

    evaluate_pipeline(conv_model)
    del conv_model
