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

file_name = "images.txt"

transform = transforms.Compose([transforms.Scale((128,128)), transforms.ToTensor()])


with open(file_name) as f:
    info = open(file_name).read().split()
    #print(info)
    for x in range(0, len(info)):
        image1 = Image.open(info[x])
        image1 = image1.convert('RGB')
        image1 = transform(image1)
        image1 = image1.unsqueeze(0)
        print(image1)
        
        
    '''
    all_names = [[None for _ in range(2)] for _ in range(len(info)/2)]
    for x in range(0,len(info)):
        all_names[x/2][x%2] = info[x]
    return all_names
    '''


