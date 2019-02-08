from handsignals.networks.torch_alexnet import AlexNet
from handsignals.networks.simple_cnn import ConvNet
from handsignals.dataset.image_dataset import ImageDataset

dataset = ImageDataset()

#a = AlexNet(dataset.num_classes())
a = ConvNet(dataset.num_classes())
a.train(dataset)
