"""This module builds pose estimation model."""

import torch
import torch.nn as nn
import constants
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
    print "Input %s x %s x 1 " % (w, h)

    o_w, o_h = w, h
    for i in range(num_downs):
        o_w, o_h = layer_calcuation(o_w, o_h, p, f, s, max_pool=False)
        o_w, o_h = layer_calcuation(o_w, o_h, p, f, s, max_pool=True)
            
    return o_w, o_h


class CNN(nn.Module):
    """This class defines the CNN Model."""

    def default_layer(self, in_channels, out_channels,
                      max_pool=True, kernel_size=3, stride=1, padding=1, dropout=0.2):
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

    def __init__(self, input_channels):

        self.input_channels = input_channels
        kernel_size = 3
        padding = 1
        stride = 1

        num_downs = int(math.log(constants.input_height)/math.log(2))
        # Since the height is greater than the width in our data we square the data
        width = height = constants.scaled_height

        o_w, o_h = layer_calculations(kernel_size, padding, stride, width, height, num_downs)

        super(CNN, self).__init__()
        # Block 1
        self.layers = nn.ModuleList()
        
        in_channels = self.input_channels
        out_channels = 1024/(2**(num_downs)) # Ensure output channels of last layer is 1024
        
        for i in range(num_downs):
            layer = self.default_layer(in_channels, out_channels, max_pool=False)
            self.layers.append(layer)
            in_channels = out_channels
            out_channels = 2 * in_channels
            layer = self.default_layer(in_channels, out_channels, max_pool=True)
            self.layers.append(layer)
            in_channels = out_channels

        self.fc = nn.Linear(out_channels * (o_w) * (o_h), 22)
        self.apply(weights_init)

    def summary(self, x):
        print x.size()
        """This is a convinence summary fuction."""
        for i, l in enumerate(self.layers):
            x = l(x)
            print x.size()

        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)

        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out
