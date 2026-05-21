import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, save_path="data/model/best_vae.pth"):
        # First epoch setup
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path)
            
        # If validation loss didn't improve by at least min_delta
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
        # If validation loss successfully improved
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path)
            self.counter = 0 # Reset the patience clock

    def save_checkpoint(self, model, save_path):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), save_path)
