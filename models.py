"""This module builds pose estimation model."""

import torch
import torch.nn as nn
import constants

w_bound = 0.005

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def conv_dim(w, h, p, f, s):
    # (n + 2p - f)/ s + 1
    o_w = (w + 2 * p - f) / s + 1
    o_h = (h + 2 * p - f) / s + 1
    return o_w, o_h


def max_pool_dim(w, h, v):
    o_w = w / v
    o_h = h / v
    return o_w, o_h


def layer_calculations(f, p, s):
    """This function calculates the required values for the model"""
    w = int(constants.default_width * constants.scale)
    h = int(constants.default_height * constants.scale)

    print "Input %s x %s  " % (w, h)

    padding_left = padding_right = padding_top = padding_bottom = 0
    if w > h:
        padding_left = padding_right = 0
        padding_top = (w - h) / 2 + (w - h) % 2
        padding_bottom = (w - h) / 2
    else:
        padding_left = (h - w) / 2 + (h - w) % 2
        padding_right = (h - w) / 2
        padding_top = padding_bottom = 0

    # print (padding_left, padding_right, padding_top, padding_bottom)

    padding_width, padding_height = w + padding_left + \
        padding_right, h + padding_top + padding_bottom
    padding_width, padding_height = padding_width, padding_height

    print "After Padding %s x %s  " % (padding_width, padding_height)

    # Layer 1
    o_w, o_h = padding_width, padding_height
    # Convolution 2D
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    # Max Pool
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 1 %s x %s x %s" % (o_w, o_h, 64)

    # Layer 2
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 2 %s x %s x %s" % (o_w, o_h, 64)

    # Layer 3
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 3 %s x %s x %s" % (o_w, o_h, 128)

    # Layer 4
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 4 %s x %s x %s " % (o_w, o_h, 256)

    # Layer 5
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 5 %s x %s x %s" % (o_w, o_h, 256)

    # Layer 6
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 6 %s x %s x %s" % (o_w, o_h, 256)

    # Layer 7
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    # o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 7 %s x %s x %s" % (o_w, o_h, 512)

    # Layer 8
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    o_w, o_h = max_pool_dim(o_w, o_h, 2)
    print "Layer 8 %s x %s x %s" % (o_w, o_h, 512)

    # Layer 9
    o_w, o_h = conv_dim(o_w, o_h, p, f, s)
    pool = o_w
    o_w, o_h = max_pool_dim(o_w, o_h, pool)
    print "Layer 9 %s x %s x %s" % (o_w, o_h, 512)

    return o_w, o_h, padding_left, padding_right, padding_top, padding_bottom, pool


class CNN(nn.Module):
    """This class defines the CNN Model."""

    def __init__(self):

        kernel_size = 3
        padding = 1
        stride = 1

        o_w, o_h, padding_left, padding_right, padding_top, padding_bottom, pool = layer_calculations(
            kernel_size, padding, stride)

        super(CNN, self).__init__()
        # Block 1
        self.layer1 = nn.Sequential(
            nn.ConstantPad2d((padding_left, padding_right,
                              padding_top, padding_bottom), 0),
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=64,  # n_filters
                      kernel_size=kernel_size,  # filter size
                      stride=stride,       # filter movement/step
                      padding=padding),      # padding
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        # Block 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        # Block 5
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Dropout(0.2)
        )
        # print pool
        self.fc = nn.Linear(512 * (o_w) * (o_h), 20)
        self.apply(weights_init)

    def summary(self):
        """This is a convinence summary fuction."""
        w = int(constants.default_width * constants.scale)
        h = int(constants.default_height * constants.scale)
        x = torch.randn(1, 1, h, w).cuda()
        print x.size()
        out = self.layer1(x)
        print out.size()
        out = self.layer2(out)
        print out.size()
        out = self.layer3(out)
        print out.size()
        out = self.layer4(out)
        print out.size()
        out = self.layer5(out)
        print out.size()
        out = self.layer6(out)
        print out.size()
        out = self.layer7(out)
        print out.size()
        out = self.layer8(out)
        print out.size()
        out = self.layer9(out)
        print out.size()
        out = out.view(out.size(0), -1)
        out = self.fc(out)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
