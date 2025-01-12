import random
import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from nnfabrik.utility.nn_helpers import set_random_seed
import sys
import modules_simple.training_lib as tl
import scipy.io
import modules_simple.data_analysis as da
#Set device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import functions and classes from the libraries
from neuralpredictors.measures.modules import PoissonLoss 
import modules_simple.Neural_Lib_Flo as nlb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


PROJECT = "New data Training, high reg, low complexity, normalized target, good_neurons, second attempt 30-07-2024"

# %%
def load_data():
    data_path = '/project/subiculum/new_data/natural_images_awake_postSub.mat'
    images_path = '/project/subiculum/new_data/new_images.npy'
    images=np.load(images_path)
    from stim_list import stim_list 
    responses, _, _,_,_ = nlb.load_mat_file(data_path)
    ids=scipy.io.loadmat('/project/subiculum/new_data/IDs.mat')
    ids=ids['rec']
    ids=ids[:,0]
    responses_processed=nlb.preprocess_responses(responses,50,120)
    stim_bool=nlb.check_for_repeated_stims(stim_list)
    test_images=images[stim_bool==1]
    test_responses=responses_processed[stim_bool==1]
    pred_array=da.oracle_prediction(test_responses,test_images,device).detach().cpu().numpy()
    good_tensor=torch.zeros(165)
    good_tensor[pred_array>0]=1
    train_loader, val_loader, test_loader = nlb.dataloader_with_repeats(responses, images, stim_list,128,cell_type=1, idx=good_tensor)
    return train_loader, val_loader, test_loader

# %%
def configure_model(config, n_neurons):
    model = nlb.DepthSepConvModel(layers=config.layers, 
                      input_kern=config.input_kern, 
                      hidden_kern=config.hidden_kern, 
                      hidden_channels=config.hidden_channels, 
                    #   spatial_scale = config.spatial_scale, 
                    #   std_scale = config.std_scale,
                      output_dim=n_neurons,
                      gaussian_readout=False,
                      reg_weight=config.reg_weight,
                      image_width=128,
                      image_height=72
                      )
    return model


def train(config, model, train_loader, val_loader, device):
    loss_0 = PoissonLoss(bias=1e-7)
    loss =lambda outputs,targets: loss_0(outputs,targets) #+ config.gamma * model.regularizer()
    # PoissonLoss(bias=1e-6) use different loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    epochs = config.epochs
    log_epoch, vcorrs, tcorrs, tlosses = [], [], [] , []
    for epoch in trange(config.epochs):
        tloss = nlb.my_train_epoch(model, train_loader, optimizer, loss, device)
        if epoch % 5 == 0:
            log_epoch.append(epoch)
            train_corrs = nlb.get_correlations(model, train_loader, device)
            val_corrs = nlb.get_correlations(model, val_loader, device)
            vcorrs.append(val_corrs.mean())
            tcorrs.append(train_corrs.mean())
            tlosses.append(tloss)
            print(f'Epoch [{epoch+1}/{epochs}], Validation correlation: {val_corrs.mean():.4f}, Training correlation: {train_corrs.mean():.4f}')
    train_corrs = nlb.get_correlations(model, train_loader, device)
    val_corrs = nlb.get_correlations(model, val_loader, device)

    return train_corrs, val_corrs, log_epoch, vcorrs, tcorrs, tlosses


def objective(config, seed=42):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()
    set_random_seed(seed)
    model = configure_model(config, n_neurons=102) 
    model.to(device)

    train_corrs, val_corrs, log_epoch, vcorrs, tcorrs, tlosses = train(config, model, train_loader, val_loader, device)
    _,_,test_corr,test_loss= tl.evaluate_model(model,test_loader,device,loss = nn.MSELoss())

    return train_corrs, val_corrs, model, log_epoch, vcorrs, tcorrs, tlosses, test_corr, test_loss

# def objective_postsub(config, seed=42):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     train_loader, val_loader, test_loader = load_data()
#     set_random_seed(seed)
#     model = configure_model(config, n_neurons=165) 
#     model.to(device)

#     train_corrs, val_corrs, log_epoch, vcorrs, tcorrs = train(config, model, train_loader, val_loader, device)

#     return train_corrs, val_corrs, model, log_epoch, vcorrs, tcorrs


def main():
    """The `main` function initializes a Weights & Biases run, runs an objective function, and logs the
    validation score. This is used for hyperparameter optimization, but it requires a Weights & Biases
    account.
    """
    with wandb.init(project=PROJECT) as wdb_run:
        train_corrs, val_corrs, model, log_epoch, vcorrs, tcorrs, tlosses,test_corr, test_loss = objective(wandb.config)
        wandb.log({"validation correlation": val_corrs.mean()})
        wandb.log({"train correlation": train_corrs.mean()})
        wandb.log({"test correlation": test_corr})
        wandb.log({"test loss": test_loss})
        for epoch, vcorr, tcorr, tloss in zip(log_epoch, vcorrs, tcorrs, tlosses):
            wandb.log({"val corr": vcorr, "epoch": epoch + 1})
            wandb.log({"train corr": tcorr, "epoch": epoch + 1})
            wandb.log({"train loss":tloss, "epoch": epoch+1})


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "name": "Gaussian Readout Hyperparameter Optimization",
    "metric": {"goal": "maximize", "name": "validation correlation"},
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 0.1},
        "layers": {"values": [1, 3, 5]},
        "epochs": {"values": [20, 30, 50]},
        "input_kern": {"values": [3, 5, 7, 11]},
        "hidden_kern": {"values": [3, 5, 7, 11]},
        "hidden_channels": {"values": [16, 32, 64]},
        # "spatial_scale": {"min": 0.05, "max": 0.3},
        # "std_scale": {"min": 0.1, "max": 1.},
        "reg_weight":{"min":0.0,"max":1e-2},
        #"weight_decay":{"min":0.0,"max":1e-2},
        #"gamma":{"min":0.0,"max":1e-3}
    },
}

if __name__ == "__main__":
    # 3: Start the sweep
    if len(sys.argv) == 1:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
    else:
        sweep_id = sys.argv[1]
    wandb.agent(sweep_id, function=main, count=50, project=PROJECT)
# %%
