import functional as F
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import random

class Square(object):
    def __call__(self, sample):
        image = np.array(sample['image'], copy=False, dtype=np.float32)
        landmarks = np.array(sample['landmarks'], copy=False, dtype=np.float32)

        image, x, y = F.square(image)

        offset = False
        if offset:
            landmarks[:, 0] += x
            landmarks[:, 1] += y        

        return {'image': image, 'landmarks': landmarks}

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.transform = transforms.Resize(size, interpolation)
        
    def __call__(self, sample):
        image = sample['image']
        sample['image'] = self.transform(image)
        return sample

class RandomShift(object):
    """Random shift the image in a sample."""

    def __init__(self, offset):
        self.offset = offset

    def __call__(self, sample):
        image = sample['image']
        landmarks = np.array(sample['landmarks'], copy=False, dtype=np.float32)

        image, tx, ty = F.shift(image, self.offset)

        landmarks = landmarks.reshape(-1, 2) 
        landmarks[:, 0] -= tx
        landmarks[:, 1] -= ty
        landmarks = landmarks.reshape(-1, 1) 

        return {'image': image, 'landmarks': landmarks}


class BlackAndWhite(object):
    """Convert image to grayscale."""

    def __call__(self, sample):
        image = np.array(sample['image'], copy=False, dtype=np.float32)
        landmarks = np.array(sample['landmarks'], copy=False, dtype=np.float32)

        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        image = np.expand_dims(image, axis=3)

        return {'image': image, 'landmarks': landmarks}

import matplotlib.pyplot as plt

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        image = np.array(sample['image'], dtype=np.float32)
        landmarks = np.array(sample['landmarks'], copy=False, dtype=np.float32)

        # plt.imshow(image/255)
        # plt.show()
        image = self.transform(image)
        landmarks = landmarks.reshape(-1, 1) 
        landmarks = torch.from_numpy(landmarks).float()
        
        return image, landmarks


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, images_mean, images_std, labels_mean, labels_std):
        self.images_mean = torch.from_numpy(images_mean)
        self.images_std = torch.from_numpy(images_std)
        self.labels_mean = torch.from_numpy(labels_mean)
        self.labels_std = torch.from_numpy(labels_std)
        self.image_transform = transforms.Normalize(self.images_mean, self.images_std)

    def __call__(self, sample):
        image, landmarks = sample
        image = self.image_transform(image)
        landmarks = (landmarks - self.labels_mean) / self.labels_std
        return image, landmarks

class RandomSwapColors(object):
    """Random swap colors of the image in a sample.
    """
    def __call__(self, sample):
        image = np.array(sample['image'], copy=False, dtype=np.float32)

        rand = random.random()
        image = F.swap_colors(image, rand)

        sample['image'] = image
        return sample

class RandomRotation(object):
    """Rotate the image by a random angle."""

    def __init__(self, degrees):
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        image = np.array(sample['image'], copy=False, dtype=np.float32)
        landmarks = np.array(sample['landmarks'], copy=False, dtype=np.float32)

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