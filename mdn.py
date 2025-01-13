# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:23:04 2024

@author: cejize
"""

import torch
import torch.nn as nn
import torch.distributions as dist

class nn_Exp(nn.Module):
    
    def forward(self, input_tensor):
        return torch.exp(input_tensor)
    


class MDN(nn.Module):
    def __init__(self, nx, n_hidden, n_gaussians, n_layers1, n_layers2, sigma_offset=0.1):
        super(MDN, self).__init__()
        self.sigma_offset = sigma_offset 
        layers = []
        layers.append(nn.Linear(nx, n_hidden))
        layers.append(nn.Tanh())
        for _ in range(n_layers1 - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        self.z_h = nn.Sequential(*layers)
        

        if n_layers2 == 1:
            self.z_pi = nn.Sequential( 
                nn.Linear(n_hidden, n_gaussians),
                nn.Softmax(dim=1)
            )
            self.z_mu = nn.Sequential( 
                nn.Linear(n_hidden, n_gaussians),
            )   
            self.z_sigma = nn.Sequential( 
                nn.Linear(n_hidden, n_gaussians),
                nn_Exp()
            )  
        else:
            layers_pi = []
            for _ in range(n_layers2-1):
                layers_pi.append(nn.Linear(n_hidden, n_hidden))
                layers_pi.append(nn.Tanh())
            self.z_pi = nn.Sequential(*layers_pi, 
                nn.Linear(n_hidden, n_gaussians),
                nn.Softmax(dim=1)
            )    
            
            
            
            layers_mu = []
            for _ in range(n_layers2-1):
                layers_mu.append(nn.Linear(n_hidden, n_hidden))
                layers_mu.append(nn.Tanh())
            self.z_mu = nn.Sequential(*layers_mu, 
                nn.Linear(n_hidden, n_gaussians),
            )    
            
            
            
            layers_sigma = []
            for _ in range(n_layers2-1):
                layers_sigma.append(nn.Linear(n_hidden, n_hidden))
                layers_sigma.append(nn.Tanh())
            self.z_sigma = nn.Sequential(*layers_mu,#layers_sigma, 
                nn.Linear(n_hidden, n_gaussians),
                nn_Exp()

            )  
      
        
    
    def forward(self, x):
        zh = self.z_h(x)
        pi = self.z_pi(zh)
        mu = self.z_mu(zh)
        sigma = self.z_sigma(zh) + self.sigma_offset 
        return pi, mu, sigma
    



def loss_fn(y, mu, sigma, pi):
    eps=1e-7
    m = dist.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(y+eps)
    prob = torch.exp(log_prob)
    weighted_prob = torch.sum(prob * pi, dim=1)  # Weighted sum of the probabilities using mixture weights
    weighted_log_prob = -torch.log(weighted_prob+eps)  # Negative log-likelihood loss


    # Total loss with entropy regularization
    loss = torch.mean(weighted_log_prob)
    return loss



def train_model(model, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    pi, mu, sigma = model(x_train)
    loss = loss_fn(y_train, mu, sigma, pi) # compute loss
    loss.backward()  # Compute gradients
    optimizer.step() # updates model parameters
    return loss.item()

def evaluate_model(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(x_val)
        loss = loss_fn(y_val, mu, sigma, pi)
    return loss.item()




