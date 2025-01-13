# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:35:36 2024

@author: cejize
"""


import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from mdn import MDN, train_model, evaluate_model
from util_eval import progressPrint, plotLoss
from bench_func import G1_samples

from scipy import stats
import random
import copy



def main_mdn(x_train, y_train, x_val, y_val, x_test, y_test, x_test_list, seed):
    
            
    # pytorch RNGs
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
    # numpy RNG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
        

    #==============================
    # Main program starts here

    epochs = 500
    nx = 1
    LR = 1e-3
    WD = 1e-3
    NH = 10
    NL = 2
    NL2 = 2
    NG = 3
    NS = 1e-3

        
    model = MDN(nx = nx, n_hidden = NH, n_gaussians = NG, n_layers1 = NL, n_layers2 = NL2, sigma_offset=NS)    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    print(f"learning_rate: {LR}, weight_decay: {WD}, n_hidden: {NH}, n_layers1: {NL}, n_layers2: {NL2}, n_gaussians: {NG}, sigma: {NS}")
    train_losses = []
    val_losses = []
        
    # Train the model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    
    for epoch in range(epochs):
        loss_train = train_model(model, optimizer, x_train, y_train)
        loss_val = evaluate_model(model, x_val, y_val)
        progressPrint(epoch, epochs, loss_train, loss_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)
            
        if loss_val < best_loss:
            best_loss = loss_val
            best_model_wts = copy.deepcopy(model.state_dict())
    
    plotLoss(epochs, train_losses, val_losses)
    
    model.load_state_dict(best_model_wts)
    model.eval()
        
        
    nll_mdn = evaluate_model(model,x_test,y_test)
    pi, mu, sigma = model(x_test_list)
        
    pi = pi.cpu()
    mu = mu.cpu()
    sigma = sigma.cpu()
        
    return pi, mu, sigma


    
if __name__ == "__main__":

    n_train = 400
    nop_train = 1
    n_val = 200
    nop_val = 1
    n_test = 200
    nop_test = 1
    seed = 0 
    x_train, y_train,_ ,_ = G1_samples(n=n_train, nop=nop_train, ptype='random')
    x_val, y_val, _ ,_ = G1_samples(n=n_val, nop=nop_val, ptype='random')
    x_test, y_test, x_test_list, y_test_array = G1_samples(n=n_test, nop=nop_test, ptype='random')
    pi, mu, sigma = main_mdn(x_train, y_train, x_val, y_val, x_test, y_test, x_test_list, seed)
    
    