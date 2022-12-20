# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 19:01:07 2022

@author: cgnya
"""

#--------Generic_Libraries---------#
import numpy as np
import time
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#---------Torch_Libraries----------#
import torch
from torchvision import models
import torch.nn as nn
from torchsummary import summary
from vit_pytorch import ViT
from vit_pytorch.max_vit import MaxViT
from torch.backends import cudnn

#------User-Defined_Libraries------#
from CUDA_data_switch import get_default_device
from EarlyStopping import EarlyStopping
device = get_default_device()

cudnn.benchmark = True

CUDA_LAUNCH_BLOCKING=1


from vit_pytorch import ViT 

class Train:
    def __init__(self,
                 optimizer,
                 loss,
                 epochs:int, 
                 patience:int,
                 modelname:str,
                 dataset:str):
        
        super(Train, self).__init__()
        self.optimizer = optimizer
        self.optimizer_name = str(optimizer)
        self.loss = loss
        self.epochs = epochs
        self.patience = patience
        self.modelname = modelname
    
    def model_(self, 
               modelname:str, 
               num_classes:int, 
               dataset:str):
        if modelname == 'vgg19':
            model = models.vgg19_bn(pretrained=True).to(device)
            model.fc = nn.Sequential(
                        nn.Linear(2048, 1024, bias=True),
                        nn.Dropout(),
                        nn.Linear(1024, 512, bias=True),
                        nn.Dropout(),
                        nn.Linear(512, num_classes, bias=True)
                        ).to(device)
            return model
        
        elif modelname == 'resnet18':
            model = models.resnet18(pretrained=False).to(device)
            model.fc = nn.Linear(512, num_classes, bias=True).to(device)
            return model
        
        elif modelname =='ViT-custom':
            model = ViT(image_size = 64,
                                patch_size = 8,
                                num_classes = 10,
                                dim = 512,
                                depth = 3,
                                heads = 8,
                                mlp_dim = 1024,
                                dropout = 0.1,
                                channels=1,
                                emb_dropout = 0.1).to('cuda:0')
            # model.fc = nn.Linear(1024, num_classes, bias=True).to('cuda:0')
            return model
        
        elif modelname == 'MaxViT':
            model = MaxViT(
                            num_classes = 1000,
                            dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
                            dim = 96,                         # dimension of first layer, doubles every layer
                            dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
                            depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
                            window_size = 7,                  # window size for block and grids
                            mbconv_expansion_rate = 4,        # expansion rate of MBConv
                            mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
                            dropout = 0.1                     # dropout
                        )
            # model.fc = nn.Linear(1024, num_classes, bias=True).to('cuda:0')
        else:
            print('ERROR_model_or_dataset_is_not_found')
      
    def top1_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return acc
    
    @torch.no_grad()
    def test(self, model, test_dl, dataset):
        model.eval()
        for batch in test_dl:
            if dataset == '\cifar10c' or dataset == '\cmnist':
                inputs, attr, q = batch
                inputs = inputs.to('cuda:0')
                attr = attr.to('cuda:0')
                labels = attr[:, 0].to('cuda:0')
            else:
                inputs, labels = batch
            outputs = model(inputs)
            loss = self.loss(outputs, labels)
            acc = self.top1_accuracy(outputs, labels)
        return loss, acc
    
    # Training Loop 
    def train(self, train_dl, test_dl, num_classes, filename, dataset):
        history = []
        since = time.time()
        es = EarlyStopping(patience=self.patience, verbose=True)
        model = self.model_(modelname=self.modelname,
                            num_classes = int(num_classes),
                            dataset = dataset)
        if self.optimizer_name == 'SGD':
            optimizer = self.optimizer(model.parameters(), lr=0.0001, momentum=0.93)
        else:
            optimizer = self.optimizer(model.parameters(), lr=0.001)
        
        for epoch in range(self.epochs):
            model.train()
            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []
            result = {}
            loop = tqdm(train_dl, total=len(train_dl))
            for batch in loop:
                if dataset == '\cifar10c' or dataset == '\cmnist':
                    p, inputs, attr, q = batch
                    inputs = inputs.to('cuda:0')
                    attr = attr.to('cuda:0')
                    labels = attr[:, 0].to('cuda:0')
                else:    
                    inputs, labels = batch
                outputs = model(inputs)
                loss = self.loss(outputs, labels)
                acc = self.top1_accuracy(outputs, labels)
                train_acc.append(acc.cpu().detach().numpy())
                train_loss.append(loss.cpu().detach().numpy())
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(train_loss=np.average(train_loss),train_acc=np.average(train_acc))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            test_losses,test_accu = self.test(model, test_dl, dataset)
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
        torch.save(model, filename + '.pth')
        torch.save(model.state_dict(), filename + 'wts.pth')
        return history
    
    
    
    
    
    