#6 incoorporate saving weights and loading them
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
from skimage import data
from skimage.transform import rotate, SimilarityTransform, warp
import random
import sys
import numpy


epochs = 35
characters = np.array(['o', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])


class loading_to_label(Dataset): # load the images without applying any random transformations, just scaling them to 128x128 and converting them to tensors (used for testing)

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
            return info
   
    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.images_name[idx])
        
        image1 = Image.open(img1_name)
        image1 = image1.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
        return image1, image1
    

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
        label_numpy = label.data.cpu().numpy()[i] ### this is a number
        #print(label_numpy)
        output_numpy = output.data.cpu().numpy()[i] # this is an array
        letter = np.argmax(output_numpy)
        truth = characters[label_numpy]
        prediction = characters[letter]
        
        if (truth == prediction): ###
            result += 1
        else:
            print(truth, prediction)
    result = result/counter
    return result

def print_result(output): # function to calculate accuracy by comparing the labels to the output of the network
    counter = 0
    np_output = output.data.cpu().numpy()
    (y, x) = np.shape(np_output)
    for i in range(0, y):
        #print(counter)
        counter += 1.0
        output_numpy = output.data.cpu().numpy()[i] # this is an array
        letter = np.argmax(output_numpy)
        if (output_numpy[letter] > 0.91):
            prediction = characters[letter]
            print(prediction)
        else:
            print("-")
    #print("counter", counter)
        


criterion = nn.CrossEntropyLoss()
cnn_model = cnn().cuda()
#optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate) # try using adam optimizer too?
#optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9) # the first test was with this
optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2, momentum=0.9)

batch_size = 12

def training(cnn_model):
    train_loss = 0
    
    dataset = image_loading(csv_file='train.txt', root_dir='LetterImages/',  transformation = transform)
    
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    train_accuracy = 0
    iterations = 0
    
    for each in dataloader: # for each pair of images loaded
        image1 = Variable(each[0]).cuda()
        #print("length", len(each[0]))
        #label1 = np.zeros((len(each[0]), 62))
        label1 = np.array([int(i) for i in each[1]])
        label = torch.from_numpy(label1).view(label1.shape[0], -1)
        label = label.view(len(label1))
        label = label.type(torch.LongTensor)
        label = Variable(label).cuda()
        #print("label", label)
        #print("image size", image1.size())
        #print("my_label size", label.size())
        output = cnn_model(image1) # get the output of the network
        optimizer.zero_grad()
        loss = criterion(output, label) # calculate the loss
        loss.backward()
        optimizer.step()
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
        label1 = np.array([int(i) for i in each[1]])
        label = torch.from_numpy(label1).view(label1.shape[0], -1)
        label = label.view(len(label1))
        label = label.type(torch.LongTensor)
        label = Variable(label).cuda()
        output = cnn_model(image1) # get the output of the network
        loss = criterion(output, label) # calculate the loss
        test_accuracy += accuracy(label, output) # calculate accuracy and add it up
        test_loss += loss.data[0]
        iterations += 1.0
    test_loss = test_loss/iterations
    test_accuracy = test_accuracy/iterations
        
    return test_loss, test_accuracy



def main():

    if len(sys.argv) != 3: # if the person didn't input an argument
        print("Usage: --load/--save WEIGHS_FILE")
        return
    
    filename = sys.argv[2] # the filename to save or load the weights

    if sys.argv[1] == "--load":
        print("loading...")
        cnn_model.load_state_dict(torch.load(filename))
        test_loss, test_accuracy = testing(cnn_model)
        train_loss, train_accuracy = training(cnn_model)
        print("train loss", round(train_loss,2))
        print("train accuracy", round(train_accuracy,2))
        print("test loss", round(test_loss,2))
        print("test accuracy", round(test_accuracy,2))

    elif sys.argv[1] == "--test":
        print("testing input images")
        cnn_model.load_state_dict(torch.load(filename))
        file_name = "images.txt"
        with open(file_name) as f:
            info = open(file_name).read().split()


        transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])

        dataset = loading_to_label(csv_file = file_name,
                                    root_dir='LetterImages/',  transformation = transform)
    
        dataloader = DataLoader(dataset, batch_size=100,
                        shuffle=False, num_workers=100)

        for each in dataloader:
            #print("once cycle of the data loader")
            image1 = Variable(each[0]).cuda()
            output = cnn_model(image1) # get the output of the network
            print_result(output)
    
        '''
            for x in range(0, len(info)):
                image1 = Image.open(info[x]) 
                image1 = image1.convert('RGB')
                image1 = transform(image1) # now I have an image
                image = Variable(image1).cuda()
                image = image.unsqueeze(0) # the image is always different, but right format?? mmm
                output = cnn_model(image)
                np_output = output.data.cpu().numpy()[0] # why is output always the same? always same image?
                print(np_output)
                letter_index = np.argmax(np_output)
                #print(letter_index)
                if (np_output[letter_index] > 0.1):
                    letter = characters[letter_index]
                    print(letter)
                else:
                    print("-")
        '''
            

        
    elif sys.argv[1] == "--save":
        print("Training... the weights will be saved at the end.")
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

        torch.save(cnn_model.state_dict(), filename)

        plt.switch_backend('agg')
        plt.plot(all_training_loss, label = "Loss")
        plt.plot(all_testing_loss, label = "Loss")
        plt.savefig('p1a_loss', bbox_inches = 'tight')

        plt.plot(all_training_accuracy, label = "accuracy")
        plt.plot(all_testing_accuracy, label = "accuracy")
        plt.savefig('p1a_accuracy', bbox_inches = 'tight')
                
    else: # if the input arguments don't match any of the options
        print("Usage: --load/--save WEIGHS_FILE")
        return
    
    return

if __name__ == '__main__':
    main()


