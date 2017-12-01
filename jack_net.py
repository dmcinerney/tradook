import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, image_size, batch_size):
        super().__init__()

        self.fc1 = nn.Linear(image_size ** 2, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
