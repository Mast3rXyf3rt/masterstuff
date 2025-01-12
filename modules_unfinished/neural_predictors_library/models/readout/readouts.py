import torch
import torch.nn as nn
from neuralpredictors.layers.readouts.factorized import FullFactorized2d

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