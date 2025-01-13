# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:01:45 2024

@author: cejize
"""
import torch
from scipy.stats import qmc
import statistics
import scipy
import numpy as np



def G1_function(x, xi1, xi2, U):
    sin_term = torch.sin(x * (xi1 + xi2)) + U
    indicator = sin_term >= -1
    G = sin_term * indicator# / 4.0
    return G

def G1_samples(n,nop,ptype):
    mu1 = 1  # Mean
    sig1 = 1  # Variance
    mu2 = 2  # Mean
    sig2 = 1  # Variance

    if ptype == 'lin':
        x = torch.linspace(0, 1, n).unsqueeze(-1)
    if ptype == 'random':
        x = torch.rand(n).unsqueeze(-1)

    z = torch.zeros(nop, len(x))
    
    
    u = torch.rand_like(z)
    x_out = x.repeat(1,nop).T
    
    xi_1 = torch.normal(mu1, torch.sqrt(torch.tensor(sig1)), size=(nop,len(x)))
    xi_2 = torch.normal(mu2, torch.sqrt(torch.tensor(sig2)), size=(nop,len(x)))
    z = G1_function(x_out, xi_1, xi_2, u)
 
    x_out = x_out.reshape(-1,1)
    z_out = z.reshape(-1,1)
    z = z.T
    return x_out,z_out,x,z



