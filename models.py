"""This module builds pose estimation model."""

import torch
import torch.nn as nn
import calculations as M
import math

w_bound = 0.005


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def layer_calcuation(o_w, o_h, p, f, s, max_pool=True):
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    if max_pool:
        o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    return o_w, o_h

def layer_calculations(f, p, s, w, h, num_downs):
    """This function calculates the required values for the model"""
    o_w, o_h = w, h
    for i in range(num_downs):
        o_w, o_h = layer_calcuation(o_w, o_h, p, f, s, max_pool=False)
        o_w, o_h = layer_calcuation(o_w, o_h, p, f, s, max_pool=True)
            
    return o_w, o_h

class Posture(nn.Module):
    """This class defines the Posture Model."""

    def default_layer(self, in_channels, out_channels, max_pool=True, kernel_size=3, stride=1, padding=1, dropout=0.2):
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        modules = []
        modules.append(conv)
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
        if max_pool:
            modules.append(nn.MaxPool2d(2))
        modules.append(nn.Dropout(dropout))

        return nn.Sequential(*modules)

    def __init__(self, input_channels, width, height, out_features):
        super(Posture, self).__init__()

        # Since the height is greater than the width in our data we square the data
                        
        self.input_channels = input_channels
        stride = 1
        kernel_size = width / 2 - 1
        padding = (stride * (width - 1) + kernel_size - width)/2

        num_downs = int(math.log(height)/math.log(2))
        num_downs = num_downs - 3
        o_w, o_h = layer_calculations(kernel_size, padding, stride, width, height, num_downs)

        # Block 1
        self.layers = nn.ModuleList()
        
        out_channels = self.input_channels
        in_channels = self.input_channels
        if num_downs > 0:
            out_channels = height/8 # Number of filters scales with image size        
            for i in range(num_downs):
                layer = self.default_layer(in_channels, out_channels, max_pool=False)
                self.layers.append(layer)
                in_channels = out_channels
                out_channels = 2 * in_channels
                layer = self.default_layer(in_channels, out_channels, max_pool=True)
                self.layers.append(layer)
                in_channels = out_channels

        self.seq = nn.Sequential(*self.layers)
        self.fc = nn.Linear(out_channels * (o_w) * (o_h), out_features)
        self.apply(weights_init)

    def summary(self, x):
        print x.size()
        """This is a convinence summary fuction."""
        for i, l in enumerate(self.layers):
            x = l(x)
            print x.size()

        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = torch.unsqueeze(out, 0)
        return out

    def forward(self, x):
        x = self.seq(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = torch.unsqueeze(out, -1)
        return out
