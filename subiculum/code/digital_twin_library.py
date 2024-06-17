from torch.utils.data import Sampler
import numpy as np

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm.notebook import tqdm
from neuralpredictors.data.datasets import FileTreeDataset
from neuralpredictors.measures.modules import PoissonLoss
from torch.utils.data import DataLoader



class SubsetSampler(Sampler):

    def __init__(self, indices, num_samples=None, shuffle=True):
        # If indices is a boolean array, convert it to an index array
        if np.issubdtype(indices.dtype, np.bool_):
            indices = np.nonzero(indices)[0]

        self.indices = indices
        if num_samples is None:
            num_samples = len(indices)
            
        self.num_samples = num_samples
        self.replace = num_samples > len(indices)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = np.random.choice(self.indices, size=self.num_samples, replace=self.replace)
        else:
            assert self.num_samples == len(self.indices), "Number of samples must be equal to the number of indices for non-shuffled sampling"
            indices = self.indices
        return iter(indices.tolist())
    
    def __repr__(self):
        return f"Random Subset Sampler on an array of {len(self.indices)}, {self.num_samples} samples per iteration and replace={self.replace}"

    def __len__(self):
        return self.num_samples
    
    
def corr(y1, y2, dim=-1, eps=1e-12, **kwargs):
    y1 = (y1 - y1.mean(axis=dim, keepdims=True)) / (y1.std(axis=dim, keepdims=True) + eps)
    y2 = (y2 - y2.mean(axis=dim, keepdims=True)) / (y2.std(axis=dim, keepdims=True) + eps)
    return (y1 * y2).mean(axis=dim, **kwargs)

class GaussianReadout(nn.Module):
    def __init__(self, output_dim, channels, spatial_scale, std_scale):
        super(GaussianReadout, self).__init__()
        self.pos_mean = nn.Parameter(torch.zeros(output_dim, 1, 2))
        self.pos_sqrt_cov = nn.Parameter(torch.zeros(output_dim, 2, 2))
        
        self.linear = nn.Parameter(torch.zeros(output_dim, channels))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self.pos_sqrt_cov.data.uniform_(-std_scale, std_scale)
        self.pos_mean.data.uniform_(-spatial_scale, spatial_scale)
        self.linear.data.fill_(1./channels)        
        
        
    def grid_positions(self, batch_size):
        if self.training:
            z = torch.randn(self.pos_mean.shape).to(self.pos_mean.device)
            grid = self.pos_mean + torch.einsum('nuk, njk->nuj', z, self.pos_sqrt_cov)
        else:
            grid = self.pos_mean
        grid = torch.clip(grid, -1, 1)
        return grid.expand(batch_size, -1, -1, -1) 
            
    def forward(self, x):
        batch_size = x.shape[0]
        grid = self.grid_positions(batch_size)
        
        # output will be batch_size x channels x neurons 
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False).squeeze(-1)
        x = torch.einsum('bcn,nc->bn', x, self.linear) + self.bias.view(1, -1)
        return x
    
class ConvModel(nn.Module):
    def __init__(self, layers, input_kern, hidden_kern, hidden_channels, output_dim, spatial_scale = 0.1, std_scale = 0.5):
        super(ConvModel, self).__init__()
        
        self.conv_layers = nn.Sequential()
        core_layers = [nn.Conv2d(1, hidden_channels, input_kern), nn.BatchNorm2d(hidden_channels), nn.SiLU()]
        
        for _ in range(layers - 1):
            core_layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, hidden_kern),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ]
            )
        self.core = nn.Sequential(*core_layers)
        
        # self.readout = FullGaussian2d((32, 18, 46), output_dim, bias=False)
        
        self.readout = GaussianReadout(output_dim, hidden_channels, spatial_scale=spatial_scale, std_scale=std_scale)
        
    def regularizer(self):
        return self.readout.linear.abs().mean()
        
    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        
        return nn.functional.softplus(x)

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for images, responses in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()
    return loss

def get_correlations(model, loader):
    """
    Calculates the correlation between the model's predictions and the actual responses.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader containing the images and responses.

    Returns:
        float: The correlation between the model's predictions and the actual responses.
    """
    resp, pred = [], []
    model.eval()
    for images, responses in loader:
        outputs = model(images)
        resp.append(responses.cpu().detach().numpy())
        pred.append(outputs.cpu().detach().numpy())
    resp = np.vstack(resp)
    pred = np.vstack(pred)
    return corr(resp, pred, dim=0)