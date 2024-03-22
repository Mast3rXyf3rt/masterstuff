# %%
import torch

import matplotlib.pyplot as plt
from tqdm import trange
import warnings
import sys
import numpy as np
warnings.filterwarnings("ignore")
import wandb
from nnfabrik.utility.nn_helpers import set_random_seed
from digital_twin_library import train_epoch, get_correlations, ConvModel, SubsetSampler
from neuralpredictors.measures.modules import PoissonLoss
from neuralpredictors.data.datasets import FileTreeDataset
from torch.utils.data import DataLoader
from neuralpredictors.data.transforms import (
    ToTensor,
    NeuroNormalizer,
    ScaleInputs,
)

PROJECT = "digital twin course 21067-10-18"

# %%
def load_data(config):
    root_dir = 'data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6'
    dat = FileTreeDataset(root_dir, 'images', 'responses')

    transforms = [ScaleInputs(scale=config.data_scale), ToTensor(torch.cuda.is_available())]
    transforms.insert(0, NeuroNormalizer(dat))
    dat.transforms.extend(transforms)


    train_sampler = SubsetSampler(dat.trial_info.tiers == 'train', shuffle=True)
    test_sampler = SubsetSampler(dat.trial_info.tiers == 'test', shuffle=False)
    val_sampler = SubsetSampler(dat.trial_info.tiers == 'validation', shuffle=False)

    train_loader = DataLoader(dat, sampler=train_sampler, batch_size=64)
    val_loader = DataLoader(dat, sampler=val_sampler, batch_size=64)
    test_loader = DataLoader(dat, sampler=test_sampler, batch_size=64)

    return dat, train_loader, val_loader, test_loader

# %%
def configure_model(config, n_neurons):
    model = ConvModel(layers=config.layers, 
                      input_kern=config.input_kern, 
                      hidden_kern=config.hidden_kern, 
                      hidden_channels=config.hidden_channels, 
                      spatial_scale = config.spatial_scale, 
                      std_scale = config.std_scale,
                      output_dim=n_neurons)
    return model


def train(config, model, train_loader, val_loader):
    loss = PoissonLoss() # use different loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    epochs = config.epochs
    log_epoch, vcorrs, tcorrs = [], [], []
    for epoch in trange(config.epochs):
        train_epoch(model, train_loader, optimizer, loss)
        if epoch % 5 == 0:
            log_epoch.append(epoch)
            train_corrs = get_correlations(model, train_loader)
            val_corrs = get_correlations(model, val_loader)
            vcorrs.append(val_corrs.mean())
            tcorrs.append(train_corrs.mean())
            
            print(f'Epoch [{epoch+1}/{epochs}], Validation correlation: {val_corrs.mean():.4f}, Training correlation: {train_corrs.mean():.4f}')
    train_corrs = get_correlations(model, train_loader)
    val_corrs = get_correlations(model, val_loader)

    return train_corrs, val_corrs, log_epoch, vcorrs, tcorrs


def objective(config, seed=42):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dat, train_loader, val_loader, test_loader = load_data(config)
    set_random_seed(seed)
    model = configure_model(config, n_neurons=dat.n_neurons)
    model.to(device)

    train_corrs, val_corrs, log_epoch, vcorrs, tcorrs = train(config, model, train_loader, val_loader)

    return train_corrs, val_corrs, model, log_epoch, vcorrs, tcorrs


def main():
    """The `main` function initializes a Weights & Biases run, runs an objective function, and logs the
    validation score. This is used for hyperparameter optimization, but it requires a Weights & Biases
    account.
    """
    with wandb.init(project=PROJECT) as wdb_run:
        train_corrs, val_corrs, model, log_epoch, vcorrs, tcorrs = objective(wandb.config)
        wandb.log({"validation correlation": val_corrs.mean()})
        wandb.log({"train correlation": train_corrs.mean()})
        for epoch, vcorr, tcorr in zip(log_epoch, vcorrs, tcorrs):
            wandb.log({"val corr": vcorr, "epoch": epoch + 1})
            wandb.log({"train corr": tcorr, "epoch": epoch + 1})


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "name": "Gaussian Readout Hyperparameter Optimization",
    "metric": {"goal": "maximize", "name": "validation correlation"},
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 0.1},
        "layers": {"values": [1, 3, 5]},
        "epochs": {"values": [20, 30, 50, 100]},
        "input_kern": {"values": [5, 7, 11]},
        "hidden_kern": {"values": [5, 7, 11]},
        "hidden_channels": {"values": [32, 64, 128]},
        "spatial_scale": {"min": 0.05, "max": 0.3},
        "std_scale": {"min": 0.1, "max": 1.},
        "data_scale": {"values": [0.25, 0.5]},
    },
}

if __name__ == "__main__":
    # 3: Start the sweep
    if len(sys.argv) == 1:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
    else:
        sweep_id = sys.argv[1]
    wandb.agent(sweep_id, function=main, count=50, project=PROJECT)