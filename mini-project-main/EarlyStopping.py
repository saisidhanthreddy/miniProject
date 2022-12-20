# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 19:33:36 2021

@author: cgnya
"""

"""
When ever the validation loss doesn't show any improvement the early 
stops helps to stop the training process after a given patience.
"""
        
import numpy as np
import torch

class EarlyStopping:
    
    def __init__(self, patience=8, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience(int): Amount of Waiting after the last validation loss improvement
                            Default: 8
            verbose(bool): If set to True, it will print a message for each validation loss improvement
                            Default: False
            path(str): Path where the checkpoint gets saved
                       Default: 'checkpoint.pth'
            trace_func(function): trace print function
                                     Default: print
            
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_score = None
        
        
    def __call__(self, val_loss, model):
        score = -val_loss        
        if self.best_score is None:
            self.best_score = score
            self.save_ckpt(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_ckpt(val_loss, model)
            self.counter = 0
        
            
    def save_ckpt(self, val_loss, model):
        """Helps in saving a model whenever there is a decrease in loss"""
        if self.verbose:
            self.trace_func(f"Validation loss decreased by ({self.val_loss_min: .6f} --> {val_loss: .6f}). \nSaving the model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
