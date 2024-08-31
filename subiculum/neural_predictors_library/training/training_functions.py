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
import torch.nn as nn
from neuralpredictors.measures.modules import PoissonLoss
from neural_predictors_library.correlation import get_correlations
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    for images, responses in loader:
        images, responses = images.to(device), responses.to(device) 
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()
    return loss

# train_epoch returns the loss of the last item in the batch, this seems very prone to errors due to outliers.

def my_train_epoch(model, loader, optimizer, loss_fn,device):
    model.train()
    train_loss = 0.0
    for images, responses in loader:
        images, responses = images.to(device), responses.to(device)  # Move data to the appropriate device
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        # # The following is for debugging (I had negative losses on the Poisson loss - was due to not normalizing my target data which had very different ranges for the neurons)
        # if loss.item() < 0:
        #     print(f"Negative loss detected: {loss.item()}")
        #     print(f"Outputs: {outputs}")
        #     print(f"Responses: {responses}")
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
    return train_loss / len(loader)




def evaluate_model(model, test_loader, device,loss):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_resps = []
    loss_fn = loss
    
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
    
    return all_preds, all_resps, correlation.mean(), test_loss

def training_and_eval_with_lr(model, epochs, train_loader, test_loader, val_loader, device, save_model= False, lr=1e-1, gamma=1e-3, path_for_saving=None, early_stopping=True, Poisson=True, weight_decay=0):
    # Define loss function and optimizer
    if Poisson == True:   
        poisson_loss = PoissonLoss(bias=1e-7)
        loss_fn = lambda outputs, targets: poisson_loss(outputs, targets) + gamma * model.regularizer()
    else:
        mse_loss = nn.MSELoss()
        loss_fn =lambda outputs,targets: mse_loss(outputs,targets) + gamma * model.regularizer()
    
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    # Define the learning rate schedule
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
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
        print("model saved as " + path_for_saving)
    # Evaluate the model
    _,_,_,_ = evaluate_model(model, test_loader, device,loss_fn)


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

