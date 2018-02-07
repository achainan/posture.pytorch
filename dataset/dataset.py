"""This module loads the dataset for the pose estimation model."""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.misc
import scipy.ndimage as ndi
import torchvision
from torchvision import transforms
import constants
from transforms import RandomShift, BlackAndWhite, ToTensor, Square, Normalize, Resize, RandomSwapColors

def train_dataset(root_dir, normalization=None, random=True, grayscale=False, square=False, csv_file='B/train_data.csv', scale=1.0):
    """This function loads the training dataset with the desired transformations."""

    scaled_width = int(round(scale * constants.default_width))

    transformations = []
    if random:
        transformations.append(RandomShift(100))
    
    if scale != 1.0:
        transformations.append(Resize(scaled_width))

    if random:
        transformations.append(RandomSwapColors())
    
    if square:
        transformations.append(Square())

    if grayscale:
        transformations.append(BlackAndWhite())

    transformations.append(ToTensor())

    if normalization is not None:
        transformations.append(normalization)

    transform = transforms.Compose(transformations)

    train_dataset = PostureLandmarksDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform)
    return train_dataset


def valid_dataset(root_dir, normalization=None, grayscale=False, square=False, csv_file='B/validation_data.csv', scale=1.0):
    """This function loads the training dataset with the desired transformations."""

    scaled_width = int(round(scale * constants.default_width))

    transformations = []
    if scale != 1.0:
        transformations.append(Resize(scaled_width))

    if square:
        transformations.append(Square())

    if grayscale:
        transformations.append(BlackAndWhite())

    transformations.append(ToTensor())

    if normalization is not None:
        transformations.append(normalization)

    transform = transforms.Compose(transformations)

    valid_dataset = PostureLandmarksDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform)
    return valid_dataset


def load_dataset(normalization, grayscale=False, root_dir='B/', csv_dir='B/', scale=1.0, random=True):
    """This function loads the datasets with the desired transformations."""

    train_data = train_dataset(root_dir, normalization=normalization, grayscale=grayscale, csv_file=csv_dir+'train_data.csv', scale=scale, random=random)
    valid_data = valid_dataset(root_dir, normalization=normalization, grayscale=grayscale, csv_file=csv_dir+'validation_data.csv', scale=scale)

    return {"train": train_data, "valid": valid_data}


class PostureLandmarksDataset(Dataset):
    """Posture Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None,
                 loader=torchvision.datasets.folder.default_loader):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        image = self.loader(img_name)

        landmarks = self.landmarks_frame.ix[idx, 1:]
        landmarks = landmarks.as_matrix().astype('float')
        item = {'image': image, 'landmarks': landmarks}

        if self.transform:
            item = self.transform(item)

        return item

