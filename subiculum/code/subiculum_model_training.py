import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Set device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import functions and classes from the libraries
from fundamental_library import *
from digital_twin_library import ConvModel, train_epoch, get_correlations, PoissonLoss

images_path = '/project/subiculum/data/images_uint8.npy'
sub_responses_path = '/project/subiculum/data/Post_Sub_Data.mat'

# Load Images to np.array
images=np.load(images_path)

# Load responses and preprocess them

sub_responses,_,_ = load_mat_file(sub_responses_path)
sub_responses = preprocess_responses(sub_responses)

#define Dataset
train_ratio=0.6
val_ratio=0.2

dataset = NeuralDataset(images, sub_responses)
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = ConvModel(
    layers=20, 
    input_kern=11,
    hidden_kern=5, 
    hidden_channels=32, 
    output_dim=37, 
    spatial_scale=0.05, 
    std_scale=0.2)

model = model.to(device)

# Define loss function and optimizer
poisson_loss = PoissonLoss()
gamma = 1e-4
loss_fn = lambda outputs, targets: poisson_loss(outputs, targets) + gamma * model.regularizer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define the learning rate schedule
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# Define early stopping criteria
early_stopping_patience = 10
early_stopping_counter = 0
best_val_loss = float('-inf')

# Define the number of epochs
epochs = 100

def my_train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    train_loss = 0.0
    for images, responses in loader:
        images, responses = images.to(device), responses.to(device)  # Move data to the appropriate device
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(loader)

for epoch in trange(epochs):
    # Training loop
    loss = my_train_epoch(model, train_loader, optimizer, loss_fn)
    
    # Validation loop
    with torch.no_grad():
        val_corrs = get_correlations(model, val_loader, device)
    validation_correlation = val_corrs.mean()
    
    # Update learning rate schedule
    lr_scheduler.step(validation_correlation)
    
    # Print training and validation losses
    print(f'Epoch [{epoch+1}/{epochs}], validation correlation: {validation_correlation:.4f}')
    
    # Check for early stopping
    if validation_correlation > best_val_loss:
        best_val_loss = validation_correlation
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print('Early stopping triggered!')
            break

# Evaluation
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

# Evaluate the model
evaluate_model(model, test_loader, device)
