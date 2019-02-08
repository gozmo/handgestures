from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from handsignals.networks.train_network import train_model
from torch import nn
import torch
import numpy as np

class ConvNet:
    def __init__(self, num_classes):
        self.cnn_model = ConvNetModel(num_classes=num_classes)
        self.cnn_model.double()


    def train(self, dataset, epochs=25):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = Adam(self.cnn_model.parameters(),
                lr=0.001)
        loss = BCEWithLogitsLoss()
        trained_model = train_model(self.cnn_model,
                dataloader,
                loss,
                optimizer,
                epochs)
        self.cnn_model, _ = trained_model

    def classify(self, image):
        image = np.asarray([image])
        image_torch = torch.from_numpy(image)
        return self.cnn_model(image_torch)

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
        self.fc = nn.Linear(153600, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
