
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from EarlyStopping import EarlyStopping
from torch.backends import cudnn


import math
import torch.utils.model_zoo as model_zoo
cudnn.benchmark = True
CUDA_LAUNCH_BLOCKING=1

from testplantdata import PlantData

PD = PlantData(batch_size=4)
#--------------Plant Dataloaders------------------#
train_dataloader, test_dataloader, num_classes = PD.plantVillage()
#---------------------------------------------------#


def top1_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    return acc

# Evaluating Loop 
@torch.no_grad()
def test(model, test_dl):
    model.eval()
    for batch in test_dl:
        inputs, labels = batch
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = top1_accuracy(outputs, labels)
    return loss, acc

# Training Loop 
def train(epochs, train_dl, test_dl, model, optimizer, patience, name):
    history = []
    since = time.time()
    optimizer = optimizer(model.parameters(), lr=0.001) 
    es = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        result = {}
        loop = tqdm(train_dl, total=len(train_dl))
        for batch in loop:
            inputs, labels = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            acc = top1_accuracy(outputs, labels)
            train_acc.append(acc.cpu().detach().numpy())
            train_loss.append(loss.cpu().detach().numpy())
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(train_loss=np.average(train_loss),train_acc=np.average(train_acc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        test_losses,test_accu = test(model, test_dl)
        test_loss.append(test_losses.cpu().detach().numpy())
        test_acc.append(test_accu.cpu().detach().numpy())       
        result['train_loss'] = np.average(train_loss)
        result['train_acc'] = np.average(train_acc)
        result['test_loss'] = np.average(test_loss)
        result['test_acc'] = np.average(test_acc)
        print('\nEpoch',epoch,result)
        history.append(result)
        tl_es = np.average(test_loss)
        es(tl_es, model)
        
        if es.early_stop:
            print("Early stopping")
            break
        print()
        
    time_elapsed = time.time() - since
    print('Training Completed in {:.0f} min {:.0f} sec'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(torch.load('checkpoint.pth'))
    torch.save(model, name + '.pth')
    torch.save(model.state_dict(), name + 'wts.pth')
    return history
            
def traintest(dataset, modelname=str, epochs=int, patience=int, i=int):
    optimizer = torch.optim.Adam
    if dataset=='plantVillage':
        if modelname == 'maxvit':
            model = models.maxvit_t(weights="DEFAULT").to('cuda')
            model.classifier = nn.Sequential(
                              nn.AdaptiveAvgPool2d(output_size=1),
                              nn.Flatten(start_dim=1, end_dim=-1),
                              nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
                              nn.Linear(in_features=512, out_features=512, bias=True),
                              nn.Tanh(),
                              nn.Linear(in_features=512, out_features=num_classes, bias=False)).to('cuda')
            print(model)
        elif modelname == 'vit':
            model = models.vit_b_32(weights="DEFAULT").to('cuda')
            model.heads = nn.Sequential(
                        nn.Linear(768, num_classes)).to('cuda')
        elif modelname == 'resnet':
            model = models.resnet18(weights='DEFAULT').to('cuda')
            model.fc = nn.Linear(512, num_classes, bias=True).to('cuda')
        elif modelname == 'vgg':
            model = models.vgg19_bn(weights="DEFAULT").to('cuda')
            model.classifier = nn.Sequential(
                        nn.Linear(25088, 4096, bias=True),
                        nn.Dropout(),
                        nn.Linear(4096, 2048, bias=True),
                        nn.Dropout(),
                        nn.Linear(2048, 1024, bias=True),
                        nn.Dropout(),
                        nn.Linear(1024, 512, bias=True),
                        nn.Dropout(),
                        nn.Linear(512, num_classes)
                        ).to('cuda')
        else:
            print('ERROR_model_or_dataset_is_not_found')
    name = dataset + modelname + '-' + str(i)
    history = train(epochs=epochs,
              train_dl=train_dataloader,
              test_dl=test_dataloader,
              model=model,
              optimizer=optimizer,
              patience = patience,
              name= name)
    df = pd.DataFrame.from_dict(history)
    df.to_csv(name, index=False)
    return df

for _ in range(3):
    traintest(dataset='plantVillage', modelname='maxvit', epochs=70, patience=25, i = _)

