import torch
import torch.nn.functional as F
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.conv_size2 = image_size * 2

        self.conv1 = nn.Conv2d(1, self.image_size, 5, padding=2)
        self.conv2 = nn.Conv2d(self.image_size, self.conv_size2, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(self.conv_size2 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 62)

        # self.fc1 = nn.Linear(image_size ** 2, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 62)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(-1, self.conv_size2 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        #
        # return F.softmax(x)

    def predict(self, x):
        logits = self.forward(x)

        return F.softmax(logits)
