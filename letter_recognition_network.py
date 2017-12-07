#7
import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import os
#import cv2
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
from skimage import data
from skimage.transform import rotate, SimilarityTransform, warp
import random
import sys
import numpy
from edge_detection_testing import *


characters = np.array(['o', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
results = []

class loading_to_label(Dataset): # load the images without applying any random transformations, just scaling them to 128x128 and converting them to tensors (used for testing)

    def __init__(self, boxes, the_image, transformation):

        self.transform = transformation
        self.boxes = boxes
        self.the_image = the_image
   
    def __len__(self):
        return len(boxes)

    def __getitem__(self, idx):
        x1 = self.boxes[idx].minX
        x2 = self.boxes[idx].maxX
        y2 = self.boxes[idx].lowY
        y1 = self.boxes[idx].highY

        image1 = self.the_image.crop((x1, y1, x2, y2))
        
        if self.transform is not None:
            image1 = self.transform(image1)
        return image1, image1
    

transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])


class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride = (1,1), padding = 2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride = (1,1), padding = 2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride = (1,1), padding = 1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride = (1,1), padding = 1)

        self.linear1 = nn.Linear(131072, 1024)
        self.linear2 = nn.Linear(1024, 62)

        self.maxPool = nn.MaxPool2d(2, stride = (2,2))
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.batch_norm5 = nn.BatchNorm1d(1024)

    def forward(self, image1):
        image1 = forward_each(self, image1) # each image goes through the same network
        results = F.sigmoid(self.linear2(image1)) # convert the results to values between 0 and 1
        
        return results

def forward_each(cnn, x):
    x = cnn.conv1(x)
    x = F.relu(x)
    x = cnn.batch_norm1(x)
    x = cnn.maxPool(x)
    x = cnn.conv2(x)
    x = F.relu(x)
    x = cnn.batch_norm2(x)
    x = cnn.maxPool(x)
    x = cnn.conv3(x)
    x = F.relu(x)
    x = cnn.batch_norm3(x)
    x = cnn.maxPool(x)
    x = cnn.conv4(x)
    x = F.relu(x)
    x = cnn.batch_norm4(x)
    x = x.view((x.data.size())[0], -1)
    x = cnn.linear1(x)
    x = F.relu(x)
    x = cnn.batch_norm5(x)

    return x


def print_result(output): # function to calculate accuracy by comparing the labels to the output of the network
    counter = 0
    np_output = output.data.cpu().numpy()
    (y, x) = np.shape(np_output)
    for i in range(0, y):
        counter += 1.0
        output_numpy = output.data.cpu().numpy()[i] # this is an array
        letter = np.argmax(output_numpy)
        if (output_numpy[letter] > 0.91):
            prediction = characters[letter]
            results.append(prediction)
        else:
            results.append('-')        

def letter_recognition_network(boxes, image_name):
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        cnn_model = cnn().cuda()
    else:
        print("running without cuda")
        cnn_model = cnn()
        
    optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2, momentum=0.9)


    filename = 'weights_40_epochs.dms'


    # cnn_model.load_state_dict(torch.load(filename), map_location=lambda storage, loc: storage)
    cnn_model.load_state_dict(torch.load(filename))

    transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])

    #boxes = get_interesting_areas('image1.jpg')
    the_image = image = Image.open(image_name)
    the_image = the_image.convert('RGB')

    dataset = loading_to_label(boxes, the_image, transformation = transform)

    dataloader = DataLoader(dataset, batch_size=12,
                    shuffle=False, num_workers=12)

    for each in dataloader:
        if torch.cuda.is_available():
            image1 = Variable(each[0]).cuda()
        else:
            image1 = Variable(each[0])
        output = cnn_model(image1) # get the output of the network
        classification = print_result(output)

    return classification 

        


