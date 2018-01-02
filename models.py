"""This module builds pose estimation model."""

import torch
import torch.nn as nn
import constants
import calculations as M

w_bound = 0.005


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def layer_calculations(f, p, s, w, h):
    """This function calculates the required values for the model"""
    print "Input %s x %s x 1 " % (w, h)

    # Layer 1
    o_w, o_h = w, h
    # Convolution 2D
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    # Max Pool
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 1 %s x %s x %s" % (o_w, o_h, 32)

    # Layer 2
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    print "Layer 2 %s x %s x %s" % (o_w, o_h, 32)

    # Layer 3
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 3 %s x %s x %s" % (o_w, o_h, 64)

    # Layer 4
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    print "Layer 4 %s x %s x %s" % (o_w, o_h, 128)

    # Layer 5
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 5 %s x %s x %s" % (o_w, o_h, 128)

    # Layer 6
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 6 %s x %s x %s " % (o_w, o_h, 256)

    # Layer 7
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 7 %s x %s x %s" % (o_w, o_h, 256)

    # Layer 8
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 8 %s x %s x %s" % (o_w, o_h, 256)

    # Layer 9
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    # o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 9 %s x %s x %s" % (o_w, o_h, 512)

    # Layer 10
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = M.max_pool_dim(o_w, o_h, 2)
    print "Layer 10 %s x %s x %s" % (o_w, o_h, 512)

    # Layer 11
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    print "Layer 11 %s x %s x %s" % (o_w, o_h, 1024)

    # Layer 12
    o_w, o_h = M.conv_dim(o_w, o_h, p, f, s)
    pool = o_w
    o_w, o_h = M.max_pool_dim(o_w, o_h, pool)
    print "Layer 12 %s x %s x %s" % (o_w, o_h, 1024)

    return o_w, o_h, pool


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

        # Since the height is greater than the width in our data we square the data
        width = height = constants.scaled_height

        o_w, o_h, pool = layer_calculations(kernel_size, padding, stride, width, height)

        super(CNN, self).__init__()
        # Block 1
        self.layers = nn.ModuleList()

        layer = self.default_layer(self.input_channels, 32, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(32, 32, max_pool=False)
        self.layers.append(layer)

        layer = self.default_layer(32, 64, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(64, 128, max_pool=False)
        self.layers.append(layer)

        layer = self.default_layer(128, 128, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(128, 256, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(256, 256, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(256, 512, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(512, 512, max_pool=False)
        self.layers.append(layer)

        layer = self.default_layer(512, 512, max_pool=True)
        self.layers.append(layer)

        layer = self.default_layer(512, 1024, max_pool=False)
        self.layers.append(layer)

        layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Dropout(0.2)
        )
        self.layers.append(layer)

        # print pool
        self.fc = nn.Linear(1024 * (o_w) * (o_h), 22)
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
