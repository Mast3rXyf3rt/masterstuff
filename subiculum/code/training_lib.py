"""
This Library contains functions for different forms of training:
- A simple training and evaluation loop
- Pretraining with different Data
- Training only the readout
- Sweep with wandb
- Oracle prediction for Data for which it makes sense
"""
import torch
import numpy as np
import scipy.io
import os
import torch.nn.functional as F
import torch.nn as nn
import warnings
from neuralpredictors.measures.modules import PoissonLoss
import seaborn as sns
import matplotlib.pyplot as plt
from Neural_Lib_Flo import *
import wandb
from tqdm.notebook import trange, tqdm


def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_resps = []
    loss_fn = nn.PoissonNLLLoss(log_input=False)
    
    with torch.no_grad():
        for images, responses in test_loader:
            images, responses = images.to(device), responses.to(device)  # Move data to the appropriate device
            outputs = model(images)
            loss = loss_fn(outputs, responses)
            test_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_resps.append(responses.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    
    all_preds = np.vstack(all_preds)
    all_resps = np.vstack(all_resps)
    correlation = get_correlations(model, test_loader, device)
    print(f'Test Correlation: {correlation.mean():.4f}')  # Print mean correlation
    
    return all_preds, all_resps

def training_and_eval_with_lr(model, epochs, train_loader, test_loader, val_loader, device, save_model= False, lr=1e-1, gamma=1e-3, path_for_saving=None, early_stopping=True):
    # Define loss function and optimizer
    poisson_loss = PoissonLoss(bias=1e-7)
    loss_fn = lambda outputs, targets: poisson_loss(outputs, targets) + gamma * model.regularizer()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Define the learning rate schedule
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('-inf')
    for epoch in trange(epochs):
        # Training loop
        loss = my_train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validation loop
        with torch.no_grad():
            val_corrs = get_correlations(model, val_loader, device)
        validation_correlation = val_corrs.mean()
        
        # Update learning rate schedule
        lr_scheduler.step(validation_correlation)
        
        # Print training and validation losses
        print(f'Epoch [{epoch+1}/{epochs}], validation correlation: {validation_correlation:.4f}, trainloss: {loss:.4f}')
        
        # Check for early stopping
        if early_stopping==True:
            if validation_correlation > best_val_loss:
                best_val_loss = validation_correlation
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print('Early stopping triggered!')
                    break
    if save_model==True:
        torch.save(model.state_dict(), path_for_saving)
        print("model saved as" + path_for_saving)
    # Evaluate the model
    evaluate_model(model, test_loader, device)


def pretraining(model, train_loader, val_loader, epochs, optimizer, loss_fn, device):
    print(f'shape in dataloader {next(iter(train_loader))[0].shape}')
    # Define early stopping criteria
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('-inf')
    pretrain_model = model.to(device)
    poisson_loss = PoissonLoss() # use different loss
    gamma = 1e-2
    #loss_fn = lambda outputs, targets: poisson_loss(outputs, targets) + gamma * pretrain_model.regularizer()
    #optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
    # Define the learning rate schedule
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    for epoch in trange(epochs):
        # Training loop
        loss = train_epoch(pretrain_model, train_loader, optimizer, loss_fn, device)
        
        # Validation loop
        with torch.no_grad():
            val_corrs = get_correlations(pretrain_model, val_loader, device)
        validation_correlation = val_corrs.mean()
        
        # Update learning rate schedule
        lr_scheduler.step(validation_correlation)
        
        # Print training and validation losses
        print(f'Epoch [{epoch+1}/{epochs}], validation correlation: {validation_correlation:.4f}, trainloss: {loss:.4f}')
        
        # Check for early stopping
        if validation_correlation > best_val_loss:
            best_val_loss = validation_correlation
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print('Early stopping triggered!')
                break

    print('Freezing core.')

    for param in pretrain_model.core.parameters():
        param.requires_grad = False

def train_readout(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, device):
    model.eval()  # Make sure the core is in eval mode
    model.readout.train()  # Only train the readout

    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('-inf')
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    for epoch in trange(num_epochs):
        model.readout.train()
        train_loss = 0.0
        for images, responses in train_loader:
            images, responses = images.to(device), responses.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, responses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0.0
        model.readout.eval()
        with torch.no_grad():
            for images, responses in val_loader:
                images, responses = images.to(device), responses.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, responses)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        with torch.no_grad():
            val_corrs = get_correlations(model, val_loader, device)
            validation_correlation = val_corrs.mean()

        lr_scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], validation correlation: {validation_correlation:.4f}, trainloss: {train_loss:.4f}')

        if validation_correlation > best_val_loss:
            best_val_loss = validation_correlation
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print('Early stopping triggered!')
                break

def oracle(model, model_state_path, device, test_loader):
    for x, y in test_loader:
        print(x.shape, y.shape)
        x = x.detach().cpu().numpy()
        print(np.abs(np.diff(x, axis=0)).max())
        break
    responses, oracle_predictor = [], []
    for _, y in test_loader:
        y = y.detach().cpu().numpy()
        responses.append(y)
        n = y.shape[0]
        trial_oracle = (n * np.mean(y, axis=0, keepdims=True) - y) / (n - 1)
        oracle_predictor.append(trial_oracle)
    responses = np.vstack(responses)
    oracle_predictor = np.vstack(oracle_predictor)
    oracle_correlation = corr(responses, oracle_predictor, dim=0)
    model= model
    state_dict=torch.load(model_state_path)
    model.load_state_dict(state_dict)
    model.to(device)
    with torch.no_grad():
        test_corrs = get_correlations(model, test_loader, device)
    sns.set_context('notebook', font_scale=1.5)
    print(test_corrs.shape)
    print(oracle_correlation.shape)
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(test_corrs, kde=False, ax = ax, color=sns.xkcd_rgb['denim blue'], label='Test')
    sns.histplot(oracle_correlation, kde=False, ax = ax, color='deeppink', label='Oracle')
    ax.legend(frameon=False)
    sns.set_context('notebook', font_scale=1.5)
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(oracle_correlation, test_corrs, s=3, color=sns.xkcd_rgb['cerulean'])
    ax.grid(True, linestyle='--', color='slategray')
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.set(
        xlabel='Oracle correlation',
        ylabel='Model correlation',
        xlim=[0, 1],
        ylim=[0, 1],
        aspect='equal',
    )

    
