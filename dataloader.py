#--------Generic_Libraries---------#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from glob import glob
from PIL import Image
import errno
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from six.moves import urllib
import gzip
import pickle


#---------Torch_Libraries----------#
import torch
from torchvision.datasets import CIFAR10, MNIST
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision.datasets import ImageFolder


#------User-Defined_Libraries------#
from CUDA_data_switch import get_default_device, DeviceDataLoader


CUDA_LAUNCH_BLOCKING=1
cudnn.benchmark = True


class LoadData:
    """
    Dataset naming convention:
        
    """
    
    def __init__(self,
                data_dir:str):
        super(LoadData, self).__init__()
        self.data_dir = data_dir
    
    
    def dataloader(self, 
                   dataset:str,
                   batch_size:int):
        
        global transforms, datasets, DataLoader, Compose, ToTensor, Resize, Normalize
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Transforms utilized for PlantVillage and PlantDoc
        traindata_transforms = Compose([
                        Resize((256, 256)),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        testdata_transforms = Compose([
                        Resize((256, 256)),  
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

        device = get_default_device()
        print(device)
          
        if self.dataset == '\plantDoc':
            num_classes = 27
            train_data = r'D:\WORK\Plant Leaf Image Disease Classification\Data\PlantDoc-Dataset-master\PlantDoc-Dataset-master\train'
            test_data = r'D:\WORK\Plant Leaf Image Disease Classification\Data\PlantDoc-Dataset-master\PlantDoc-Dataset-master\test'
            train_dataset = datasets.ImageFolder(os.path.join(train_data), traindata_transforms)
            test_dataset = datasets.ImageFolder(os.path.join(test_data), testdata_transforms)
            train_dl = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            test_dl = DataLoader(test_dataset, self.batch_size, shuffle=True) 
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dataloader, test_dataloader, num_classes
        
        elif self.dataset == '\plantVillage':
            num_classes = 38
            def train_val_dataset(dataset, val_split=0.20):
                train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
                datasets = {}
                datasets['train'] = Subset(dataset, train_idx)
                datasets['test'] = Subset(dataset, val_idx)
                return datasets
            
            dataset = ImageFolder(r'D:\WORK\Plant Leaf Image Disease Classification\Data\PlantVillage-Dataset-master\PlantVillage-Dataset-master\raw\color', 
                                  transform=Compose([Resize((256,256)),
                                                     ToTensor(),
                                                     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
            datasets = train_val_dataset(dataset)
            train_dataset =datasets['train']
            test_dataset = datasets['test']
            train_dl = DataLoader(datasets['train'], self.batch_size, shuffle=True) 
            test_dl = DataLoader(datasets['test'], self.batch_size, shuffle=True) 
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dataloader, test_dataloader, num_classes

        else:
            print('Invalid_Entry')
            
