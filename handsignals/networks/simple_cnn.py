from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

from handsignals.networks.base_network import BaseNetwork
from handsignals.networks.train_network import train_model
from torch import nn
import torch
import numpy as np
from handsignals import device

class ConvNet(BaseNetwork):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.__model = ConvNetModel(num_classes=num_classes)
        super().__init__(self.__model)

    def train(self,
              training_dataset,
              holdout_dataset,
              model_parameters):
        optimizer = Adam(self.__model.parameters(),
                         lr=model_parameters.learning_rate)
        loss = BCEWithLogitsLoss(reduce="sum")
        trained_model, validation_loss, training_loss= self.train_model(model_parameters,
                                                                   training_dataset,
                                                                   holdout_dataset,
                                                                   loss,
                                                                   optimizer)
        self.__model = trained_model
        return validation_loss, training_loss

    def load(self, path):
        self.__model = ConvNetModel(self.num_classes)
        self.__model.load_state_dict(torch.load(path))
        self.__model.double()
        self.__model.to(device)

class ConvNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNetModel, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(9600, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        self.to(device)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
