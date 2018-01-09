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
import functional as F

def train_dataset(root_dir, normalization=None, random=True, grayscale=True, csv_file='B/train_data.csv', scale=1.0):
    """This function loads the training dataset with the desired transformations."""

    transformations = [Scale(scale)]
    if random:
        transformations.append(RandomHorizontalFlip())
        transformations.append(RandomShift(100 * scale))
        transformations.append(RandomSwapColors())

    transformations.append(Square())

    if grayscale:
        transformations.append(BlackAndWhite())

    if normalization is not None:
        transformations.append(normalization)

    transformations.append(ToTensor())

    transform = transforms.Compose(transformations)

    train_dataset = PostureLandmarksDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform)
    return train_dataset


def valid_dataset(root_dir, normalization=None, grayscale=True, csv_file='B/validation_data.csv', scale=1.0):
    """This function loads the training dataset with the desired transformations."""

    transformations = [Scale(scale)]

    transformations.append(Square())

    if grayscale:
        transformations.append(BlackAndWhite())

    if normalization is not None:
        transformations.append(normalization)

    transformations.append(ToTensor())

    transform = transforms.Compose(transformations)

    valid_dataset = PostureLandmarksDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform)
    return valid_dataset


def load_dataset(normalization, grayscale=True, root_dir='B/', csv_dir='B/', scale=1.0):
    """This function loads the datasets with the desired transformations."""

    train_data = train_dataset(root_dir, normalization=normalization, grayscale=grayscale, csv_file=csv_dir+'train_data.csv', scale=scale)
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
        landmarks = landmarks.reshape(-1, 2)
        item = {'image': image, 'landmarks': landmarks}

        if self.transform:
            item = self.transform(item)

        return item

class Square(object):
    """Pad the image to make a square in a sample."""

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)

        image, x, y = F.square(image)

        landmarks[:, 0] += x
        landmarks[:, 1] += y        

        return {'image': image, 'landmarks': landmarks}


class Scale(object):
    """Scale the image in a sample."""

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)

        image = scipy.misc.imresize(image, self.size)
        landmarks *= self.size

        return {'image': image, 'landmarks': landmarks}


class RandomShift(object):
    """Random shift the image in a sample."""

    def __init__(self, offset):
        self.offset = offset

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)

        image, tx, ty = F.shift(image, self.offset)
        
        landmarks[:, 0] -= tx
        landmarks[:, 1] -= ty

        return {'image': image, 'landmarks': landmarks}


class RandomHorizontalFlip(object):
    """Flip horizontally the image in a sample."""

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)
        # image, landmarks = sample['image'], sample['landmarks']

        rand = np.random.uniform(0, 1)
        if rand > 0.5:
            _, w = image.shape[:2]

            axis = 1
            image = np.asarray(image).swapaxes(axis, 0)
            image = image[::-1, ...]
            image = image.swapaxes(0, axis)

            landmarks[1], landmarks[0] = landmarks[0], landmarks[1].copy()
            landmarks[3], landmarks[2] = landmarks[2], landmarks[3].copy()
            landmarks[5], landmarks[4] = landmarks[4], landmarks[5].copy()
            landmarks[7], landmarks[6] = landmarks[6], landmarks[7].copy()
            landmarks[9], landmarks[8] = landmarks[8], landmarks[9].copy()

            landmarks[:, 0] *= -1
            landmarks = landmarks + [w, 0]

        return {'image': image, 'landmarks': landmarks}


class BlackAndWhite(object):
    """Convert image to grayscale."""

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)

        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        image = np.expand_dims(image, axis=3)

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=False, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)

        image = F.image_to_tensor(image)
        return image, torch.from_numpy(landmarks).float()


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, images_mean, images_std, labels_mean, labels_std):
        self.images_mean = images_mean
        self.images_std = images_std
        self.labels_mean = labels_mean
        self.labels_std = labels_std

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)

        image = (image - self.images_mean) / self.images_std

        landmarks = (landmarks - self.labels_mean) / self.labels_std

        return {'image': image, 'landmarks': landmarks}

class RandomSwapColors(object):
    """Random swap colors of the image in a sample.
    """
    def __call__(self, sample):
        image = np.array(sample['image'], copy=True, dtype=np.float32)
        landmarks = np.array(sample['landmarks'], copy=True, dtype=np.float32)

        rand = np.random.uniform(0, 1)
        image = F.swap_colors(image, rand)

        return {'image': image, 'landmarks': landmarks}

class RandomRotation(object):
    """Rotate the image by a random angle."""

    def __init__(self, degrees):
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        image = np.array(sample['image'], copy=True, dtype=np.float32)
        landmarks = np.array(sample['landmarks'], copy=True, dtype=np.float32)

        angle = self.get_params(self.degrees)

        # Rotate the image
        image = ndi.rotate(image, angle, cval=255, reshape=False)

        # Rotate the landmarks
        org_center = (np.array(image.shape[:2][::-1]) - 1) / 2.
        org = np.zeros(landmarks.shape)
        org[:, 0] = landmarks[:, 0] - org_center[0]
        org[:, 1] = landmarks[:, 1] - org_center[1]
        a = np.deg2rad(angle)
        new = np.zeros(landmarks.shape)
        new[:, 0] = org[:, 0] * np.cos(a) + org[:, 1] * np.sin(a)
        new[:, 1] = -org[:, 0] * np.sin(a) + org[:, 1] * np.cos(a)
        landmarks[:, 0] = new[:, 0] + org_center[0]
        landmarks[:, 1] = new[:, 1] + org_center[1]

        return {'image': image, 'landmarks': landmarks}
