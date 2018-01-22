import torch
import torch.nn as nn

import numpy as np

import dataset

def normalization_values(grayscale=True, flatten=True, root_dir='B/', csv_file='B/train_data.csv', scale=1.0):
    batch_size = sum(1 for line in open(csv_file))
    batch_size = batch_size - 1
    
    train_dataset = dataset.train_dataset(root_dir=root_dir, csv_file=csv_file, random=False, grayscale=grayscale, scale=scale)
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size, # Total number of training samples
                                             shuffle=False,
                                             drop_last=True)

    images, labels = iter(dataloader).next()

    numpy_images = images.numpy()
    numpy_labels = labels.numpy()

    images_mean = np.mean(numpy_images, axis=(0, )).transpose((1, 2, 0))
    images_std = np.std(numpy_images, axis=(0, )).transpose((1, 2, 0))

    labels_mean = np.mean(numpy_labels, axis=(0))
    labels_std = np.std(numpy_labels, axis=(0))

    if flatten:
        images_mean = np.mean(numpy_images, axis=(0, 2, 3))
        images_std = np.std(numpy_images, axis=(0, 2, 3))

        labels_mean = np.mean(numpy_labels, axis=(0, 1, 2))
        labels_std = np.std(numpy_labels, axis=(0, 1, 2))

    return images_mean, images_std, labels_mean, labels_std
