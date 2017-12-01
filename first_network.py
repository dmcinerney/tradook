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
import cv2
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
#ImageFile = LetterImages
#ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import data
from skimage.transform import rotate, SimilarityTransform, warp
import random
import sys
import numpy

class image_loading(Dataset): # load the images without applying any random transformations, just scaling them to 128x128 and converting them to tensors (used for testing)

    def __init__(self, csv_file, root_dir, transformation):

        self.root_dir = root_dir
        self.transform = transformation
        self.images_name = self.read_each_name(csv_file)

    def read_each_name(self, file_name):
        with open(file_name) as f:
            info = open(file_name).read().split()
            all_names = [[None for _ in range(2)] for _ in range(len(info)/2)]
            for x in range(0,len(info)):
                all_names[x/2][x%2] = info[x]
            return all_names
   
    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.images_name[idx][0])
        label = self.images_name[idx][1]
        
        image1 = Image.open(img1_name)
        image1 = image1.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
        return image1, label


transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])

dataset = image_loading(csv_file='train.txt',
                                    root_dir='LetterImages/',  transformation = transform)
    
dataloader = DataLoader(dataset, batch_size=12,
                        shuffle=True, num_workers=12)



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
        #print(image1)
        #both_results = torch.cat((image1, image2), 1) # combine the results of both images
        #results = self.linear2(image1)
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


def accuracy(label, output): # function to calculate accuracy by comparing the labels to the output of the network
    result = 0
    counter = 0
    np_output = output.data.cpu().numpy()
    (y, x) = np.shape(np_output)
    for i in range(0, y):
        counter += 1.0
        label_numpy = label.data.cpu().numpy()[i]
        output_numpy = output.data.cpu().numpy()[i]
        letter = np.argmax(output_numpy)
        if (label_numpy[letter] == 1.0):
            result += 1
    result = result/counter
    return result

learning_rate = 1e-6
criterion=nn.MultiLabelMarginLoss()
#criterion = nn.BCELoss()
cnn_model = cnn().cuda()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
batch_size = 12

def training(cnn_model):
    train_loss = 0
    
    dataset = image_loading(csv_file='train.txt', root_dir='LetterImages/',  transformation = transform)
    
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    train_accuracy = 0
    iterations = 0
    
    for each in dataloader: # for each pair of images loaded
        image1 = Variable(each[0]).cuda()
        label1 = np.zeros((batch_size, 62))
        label_identifier = np.array([int(i) for i in each[1]])
        for x in range(0, len(label_identifier)):
            label1[x][label_identifier[x]] = 1
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.LongTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1) # get the output of the network
        optimizer.zero_grad()
        loss = criterion(output, label) # calculate the loss
        loss.backward()
        optimizer.step()
        #output = torch.round(output) # round to 0 and 1 in order to compare the output to the labels
        train_accuracy += accuracy(label, output) # calculate accuracy and add it up
        train_loss += loss.data[0]
        iterations += 1.0
    train_loss = train_loss/iterations
    train_accuracy = train_accuracy/iterations


    return train_loss, train_accuracy


def testing(cnn_model):
    dataset = image_loading(csv_file='test.txt', root_dir='LetterImages/',  transformation = transform)
    
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)
    
    test_loss = 0
    test_accuracy = 0
    iterations = 0

    for each in dataloader: # for each pair of images loaded
        image1 = Variable(each[0]).cuda()
        label1 = np.zeros((batch_size, 62))
        label_identifier = np.array([int(i) for i in each[1]])
        for x in range(0, len(label_identifier)):
            label1[x][label_identifier[x]] = 1
        label1 = torch.from_numpy(label1).view(label1.shape[0], -1)
        label1 = label1.type(torch.LongTensor)
        label = Variable(label1).cuda()
        output = cnn_model(image1) # get the output of the network
        loss = criterion(output, label) # calculate the loss
        train_accuracy += accuracy(label, output) # calculate accuracy and add it up
        train_loss += loss.data[0]
        iterations += 1.0
    train_loss = train_loss/iterations
    train_accuracy = train_accuracy/iterations
        
    return test_loss, test_accuracy


epochs = 5

all_training_loss = list()
all_testing_loss = list()
all_training_accuracy = list()
all_testing_accuracy = list()

for epoch in range(epochs):
    print("epoch", epoch)
    train_loss, train_accuracy = training(cnn_model)
    print("train loss", train_loss)
    print("train accuracy", train_accuracy)
    all_training_loss.append(train_loss)
    all_training_accuracy.append(train_accuracy)
    test_loss, test_accuracy = testing(cnn_model)
    print("test loss", test_loss)
    print("test accuracy", test_accuracy)
    all_testing_loss.append(test_loss)
    all_testing_accuracy.append(test_accuracy)

print("training loss ", all_training_loss)
print("testing loss", all_testing_loss)
print("training accuracy", all_training_accuracy)
print("testing accuracy", all_testing_accuracy)



