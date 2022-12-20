# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:06:57 2022

@author: cgnya
"""
import torch.optim as optim
import pandas as pd
from dataloader import LoadData
from Train import Train
from lossfunctions import LossFunctions
batch_size = 4
optimizer = optim.Adam
epochs = 70
patience = 12
modelname = 'resnet18'

# Loading Data into Dataloaders
directory = r'E:\Analysis_Work\data'
dl = LoadData(directory)

# for _ in range(1):
#     for dataset in ('\plantVillage','\plantDoc'):
dataset = '\plantDoc'
train_dataloader, test_dataloader, num_classes = dl.dataloader(dataset, batch_size)

# Declaring Loss Function
l = LossFunctions(num_classes)

# Training 
t = Train(optimizer = optimizer,
          loss = l.cross_entropy_loss,
          epochs = epochs,
          patience = patience,
          modelname = modelname,
          dataset = dataset,
          )
print(dataset)
filename = dataset[1:] + modelname

history = t.train(
            train_dl = train_dataloader,
            test_dl = test_dataloader,
            num_classes = num_classes,
            filename = filename,
            dataset = dataset)

# Saving
df = pd.DataFrame.from_dict(history)
df.to_csv('{}.csv'.format(filename),sep=str(','))