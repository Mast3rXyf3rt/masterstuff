import numpy as np
import torch.nn as nn
from models.readout.readouts import GaussianReadout
from models.readout.readouts import FullFactorized2d



def readout_input(layers, input_kern, hidden_kern, image_width, image_height):
    input1=(image_width-input_kern+5)-(layers-1)*(hidden_kern-5)
    input2=(image_height-input_kern+5)-(layers-1)*(hidden_kern-5)
    return input1, input2
class ConvModel(nn.Module):
    def __init__(self, layers, input_kern, hidden_kern, hidden_channels, output_dim, spatial_scale = 0.1, std_scale = 0.5, gaussian_readout=True, reg_weight=None, image_width=64, image_height=64):
        super(ConvModel, self).__init__()
        
        self.conv_layers = nn.Sequential()
        core_layers = [nn.Conv2d(1, hidden_channels, input_kern, padding=2), nn.BatchNorm2d(hidden_channels), nn.SiLU()]
        
        for _ in range(layers - 1):
            core_layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, hidden_kern, padding=2),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ]
            )
        self.core = nn.Sequential(*core_layers)
        
        # self.readout = FullGaussian2d((32, 18, 46), output_dim, bias=False)
        if gaussian_readout ==True:
            self.readout = GaussianReadout(output_dim, hidden_channels, spatial_scale=spatial_scale, std_scale=std_scale)
        else:
            self.readout = FullFactorized2d(
                in_shape=(hidden_channels,)+ readout_input(layers, input_kern, hidden_kern, image_width, image_height),  # Set the appropriate shape
                outdims=output_dim,
                bias=True,
                spatial_and_feature_reg_weight=reg_weight or 1.0)
    
    def regularizer(self):
        if isinstance(self.readout, FullFactorized2d):
            return self.readout.regularizer()
        return self.readout.linear.abs().mean()
        
    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        
        return nn.functional.softplus(x)
    
# Depthwise Model:
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(n_in, n_in, kernel_size, padding=padding, groups=n_in)
        self.pointwise = nn.Conv2d(n_in, n_out, kernel_size= 1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class DepthSepConvModel(nn.Module):
    def __init__(self, layers, input_kern, hidden_kern, hidden_channels, output_dim, spatial_scale = 0.1, std_scale = 0.5, gaussian_readout=True, reg_weight=None,image_width=64, image_height=64):
        super(DepthSepConvModel, self).__init__()
        
        self.conv_layers = nn.Sequential
        core_layers = [nn.Conv2d(1, hidden_channels, input_kern, padding=2), nn.BatchNorm2d(hidden_channels), nn.SiLU()]
        
        for _ in range(layers - 1):
            core_layers.extend([
                DepthwiseSeparableConv(hidden_channels, hidden_channels, hidden_kern, padding=2),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ]
            )
        self.core = nn.Sequential(*core_layers)
        if gaussian_readout ==True:
            self.readout = GaussianReadout(output_dim, hidden_channels, spatial_scale=spatial_scale, std_scale=std_scale)
        else:
            self.readout = FullFactorized2d(
                in_shape=(hidden_channels,)+ readout_input(layers, input_kern, hidden_kern, image_width, image_height),  # Set the appropriate shape
                outdims=output_dim,
                bias=True,
                spatial_and_feature_reg_weight=reg_weight or 1.0)
    
    def regularizer(self):
        if isinstance(self.readout, FullFactorized2d):
            return self.readout.regularizer()
        return self.readout.linear.abs().mean()
        
    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        return nn.functional.softplus(x)
