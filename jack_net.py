import torch
import torch.nn.functional as F
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, image_size):
        super(ConvNet, self).__init__()
        self.image_size = image_size
        self.conv_size2 = 64 * 2

        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, self.conv_size2, 5, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 12 * 12, 100) #128 * 12 * 12 is size (2, 3,4 size indices) after second relu
        self.fc2 = nn.Linear(100, 62)

        # self.fc1 = nn.Linear(image_size ** 2, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        # print 'after conv1', x.size()
        x = self.pool(x)
        # print 'after pool', x.size()
        x = F.relu(x)
        # print 'after relu', x.size()
        # print
        x = self.conv2(x)
        # print 'after conv2', x.size()
        x = self.pool(x)
        # print 'after pool', x.size()
        x = F.relu(x)
        # print 'after relu', x.size()

        # x = F.relu(self.pool(self.conv2(x)))
        # print 'after conv2', x.size()
        # x = x.view(-1, self.conv_size2 * 7 * 7)
        x = x.view(-1, self.num_flat_features(x))
        # print 'after view', x.size()

        x = F.relu(self.fc1(x))
        # print 'after last relu'
        x = self.fc2(x)
        # print 'after fc2'
        # print x.size()

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

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
