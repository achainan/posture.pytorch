"""This module loads the dataset for the pose estimation model."""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.misc
import torchvision
from torchvision import transforms
import constants
from third_party import apply_transform

images_mean, images_std, labels_mean, labels_std = 216.91674805, 51.54261398, 147.78466797, 57.77311325


def train_dataset(normalization=None, random=True):
    transformations = [Scale(constants.scale)]
    if random:
        transformations.append(RandomHorizontalFlip())
        transformations.append(RandomShift(100 * constants.scale))

    transformations.append(BlackAndWhite())

    if normalization is not None:
        transformations.append(normalization)

    transformations.append(ToTensor())

    transform = transforms.Compose(transformations)

    train_dataset = PostureLandmarksDataset(
        csv_file='B/train_data.csv',
        root_dir='B/',
        transform=transform)
    return train_dataset


def load_dataset():
    """This function loads the dataset with the desired transformations."""

    normalization = Normalize(images_mean, images_std, labels_mean, labels_std)
    train_data = train_dataset(normalization)

    valid_data = PostureLandmarksDataset(csv_file='B/validation_data.csv',
                                         root_dir='B/',
                                         transform=transforms.Compose([
                                             Scale(constants.scale),
                                             BlackAndWhite(),
                                             normalization,
                                             ToTensor()
                                         ]))

    return {"train": train_data, "valid": valid_data}


class PostureLandmarksDataset(Dataset):
    """Posture Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None,
                 loader=torchvision.datasets.folder.default_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        # image = io.imread(img_name)
        image = self.loader(img_name)

        landmarks = self.landmarks_frame.ix[idx,
                                            1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Scale(object):
    """Random shift the image in a sample.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, landmarks = np.array(sample['image'], copy=True, dtype=np.float32), np.array(
            sample['landmarks'], copy=True, dtype=np.float32)
        # image, landmarks = sample['image'], sample['landmarks']

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

        channel_axis = 2
        cval = 0.
        tx = np.random.randint(-1 * self.offset, high=self.offset)
        ty = 0  # Currently only horizontal shift
        translation_matrix = np.array([[1, 0, ty],
                                       [0, 1, tx],
                                       [0, 0, 1]])

        transform_matrix = translation_matrix  # no need to do offset
        image = apply_transform(image, transform_matrix,
                                channel_axis, 'nearest', cval)

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

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        landmarks = landmarks.reshape(1, -1)
        return image.float(), torch.from_numpy(landmarks).float()


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
