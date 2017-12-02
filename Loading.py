import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
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
        # print type(label)
        image1 = Image.open(img1_name)
        image1 = image1.convert('RGB')
        # print type(image1)
        # print image1.size
        if self.transform is not None:
            image1 = self.transform(image1)
        return image1, label


# transform = transforms.Compose([transforms.Scale((64,64)), transforms.ToTensor()])
#
# dataset = image_loading(csv_file='train.txt',
#                                     root_dir='LetterImages/',  transformation = transform)
#
# dataloader = DataLoader(dataset, batch_size=12,
#                         shuffle=True, num_workers=12)

def show_batch(sample_batch):
    images_batch1 = sample_batch[0]
    images_batch2 = sample_batch[1]

    grid1 = utils.make_grid(images_batch1).numpy().transpose((1, 2, 0))
    # print grid1.shape
    plt.imshow(grid1)
    plt.axis('off')
    plt.ioff()
    plt.show()

    grid2 = utils.make_grid(images_batch2).numpy().transpose((1, 2, 0))
    plt.imshow(grid2)
    plt.axis('off')
    plt.ioff()
    plt.show()
#
# print(dataloader)
# for sample_batch in dataloader:
#     # x = 0
#     # print sample_batch.size()
#     plt.figure()
#     show_batch(sample_batch)
