#Array, image processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Model Operation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor , CenterCrop
from torch.utils.data import DataLoader , Dataset
from torchsummary import summary
from torchviz import make_dot

from sklearn.model_selection import train_test_split
# io
import os
#import opendatasets as od
import pandas as pd
import glob
import pathlib
from tqdm import tqdm
import warnings;
warnings.filterwarnings('ignore')

IMG_SIZE = 512

def plot_subdata(dataset , subfolders) :
    for sub in subfolders :
        imgfolder_path = os.path.join(dataset , sub , sub , 'CameraRGB')
        maskfolder_path = os.path.join(dataset , sub , sub , 'CameraSeg')

        img_files = os.listdir(imgfolder_path)
        mask_files = os.listdir(maskfolder_path)

        for i in range(3) :
            img_path = os.path.join(imgfolder_path , img_files[i])
            mask_path = os.path.join(maskfolder_path , mask_files[i])

            img = cv2.cvtColor(cv2.imread(img_path) , cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Image {i+1} - Subfolder: {sub}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.title(f'Mask {i+1} - Subfolder: {sub}')
            plt.axis('off')

            plt.show()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)
    
class CopyAndCrop(nn.Module):
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        _, _, h, w = skip_connection.shape
        crop = CenterCrop((h, w))(x)
        residual = torch.cat((x, crop), dim=1)

        return residual

class UNET(nn.Module) :
    def __init__(self , in_channels, out_channels) :

        super().__init__()

        self.encoders = nn.ModuleList([
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        ])

        self.pool = nn.MaxPool2d(2)

        self.copyAndCrop = CopyAndCrop()

        self.decoders = nn.ModuleList([
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        ])

        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

        self.bottleneck = ConvBlock(512, 1024)

        self.finalconv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)


    def forward(self , x ) :

        skip_connections = []

        # Encoding

        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoding

        for idx, dec in enumerate(self.decoders):
            x = self.up_samples[idx](x)
            skip_connection = skip_connections.pop()
            x = self.copyAndCrop(x, skip_connection)
            x = dec(x)
        x = self.finalconv(x)
        return x

class CustomDataset(Dataset):
    def __init__(self,img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize(size = (IMG_SIZE , IMG_SIZE)) ,
            transforms.RandomRotation(20) ,
            transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomCrop(460),
            ToTensor()

        ])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = Image.open(self.img_dir[idx])
        tensor_image = self.transform(image)
        mask = Image.open(self.mask_dir[idx])
        tensor_mask = self.transform(mask)

        return tensor_image , tensor_mask