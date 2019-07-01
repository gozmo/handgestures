from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from handsignals.networks.train_network import train_model
from torch import nn
import torch


class AlexNet:
    def __init__(self, num_classes):
        print("init")
        self.__alexnet_model = AlexNetModel(num_classes=num_classes)
        self.__alexnet_model.double()

    def train(self, dataset, epochs=2):
        print("train")
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = Adam(self.__alexnet_model.parameters(), lr=0.001)
        loss = CrossEntropyLoss
        trained_model = train_model(
            self.__alexnet_model, dataloader, loss, optimizer, epochs
        )
        self.__alexnet_model = trained_model

    def classify(self, image):
        pass


class AlexNetModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        print(x)
        x = self.features(x)
        print(x)
        x = x.view(x.size(0), 4096)
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        print(x)
        return x
