# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:55:04 2024

@author: cejize
"""
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def progressPrint(i, n, loss_train, loss_val):
    progress = ((i + 1) / n) * 100
    if progress % 10 == 0:
        print('Training completed {:.0f}%'.format(progress), end=', ')
        #print(f"Epoch: {i}, Loss: {loss}")
        print(f"Train Loss: {loss_train}, Val Loss: {loss_val}")

# Plot the training and validation loss
def plotLoss(epochs, train_losses, val_losses):
    plt.figure(12000)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss-Epoch Curve')
    plt.legend()
    plt.show()