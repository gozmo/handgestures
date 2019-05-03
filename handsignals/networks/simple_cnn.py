from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from handsignals.networks.train_network import train_model
from torch import nn
import torch
import numpy as np
from handsignals import device

class ConvNet:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.cnn_model = ConvNetModel(num_classes=num_classes)
        self.cnn_model.double()
        self.cnn_model.to(device)

    def train(self, 
              dataset, 
              epochs,
              batch_size,
              learning_rate):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.cnn_model.parameters(),
                lr=learning_rate)
        loss = BCEWithLogitsLoss(reduce="sum")
        self.cnn_model.to(device)
        trained_model, validation_loss, training_loss= train_model(self.cnn_model,
                dataloader,
                loss,
                optimizer,
                epochs)
        self.cnn_model = trained_model
        return validation_loss, training_loss

    def classify(self, image):
        image = np.asarray([image])
        image_torch = torch.from_numpy(image)
        image_torch = image_torch.to(device)
        return self.cnn_model(image_torch)

    def classify_batch(self, batch):
        images = np.asarray(batch)
        images_torch = torch.from_numpy(images)
        images_torch = images_torch.to(device)
        return self.cnn_model(images_torch)


    def save(self, path):
        torch.save(self.cnn_model.state_dict(), path)

    def load(self, path):
        self.cnn_model = ConvNetModel(self.num_classes)
        self.cnn_model.load_state_dict(torch.load(path))
        self.cnn_model.double()
        self.cnn_model.to(device)

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
