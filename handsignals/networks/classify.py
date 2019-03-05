from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset
from handsignals.networks.simple_cnn import ConvNet

def classify(dataset):

    labels = dataset.num_classes()
    conv_model = ConvNet(labels)
    conv_model.load

    predictions = []
    for idx in range(len(dataset)):
        image = dataset[idx]["image"]
        #prediction = conv_model.classify(image)
        prediction = "mock_label"
        predictions.append(prediction)
    return predictions
