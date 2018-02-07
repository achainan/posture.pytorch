import torch
import torch.nn as nn

import numpy as np

import dataset

# If we want a single variables for the mean and standard deviation values we set flatten_images and flatten_labels to true.

def normalization_values(grayscale=False, flatten_images=True, flatten_labels=True, root_dir='B/', csv_file='B/train_data.csv', scale=1.0, cache=False):
    if cache:
        images_mean = np.loadtxt('image_mean.txt', dtype=np.float32)
        images_std = np.loadtxt('image_std.txt', dtype=np.float32)
        labels_mean = np.loadtxt('label_mean.txt', dtype=np.float32).reshape(-1, 1)
        labels_std = np.loadtxt('label_std.txt', dtype=np.float32).reshape(-1, 1)
    else:    
        batch_size = sum(1 for line in open(csv_file))
        batch_size = batch_size - 1
    
        train_dataset = dataset.train_dataset(root_dir=root_dir, csv_file=csv_file, random=False, grayscale=grayscale, scale=scale)
        dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        images, labels = iter(dataloader).next()

        numpy_images = images.numpy()
        images_mean = np.mean(numpy_images, axis=(0, )).transpose((1, 2, 0))
        images_std = np.std(numpy_images, axis=(0, )).transpose((1, 2, 0))
        if flatten_images:
            images_mean = np.mean(numpy_images, axis=(0, 2, 3))
            images_std = np.std(numpy_images, axis=(0, 2, 3))

        numpy_labels = labels.numpy()
        labels_mean = np.mean(numpy_labels, axis=(0))
        labels_std = np.std(numpy_labels, axis=(0))
        if flatten_labels:
            labels_mean = np.array([np.mean(numpy_labels, axis=(0, 1, 2))])
            labels_std = np.array([np.std(numpy_labels, axis=(0, 1, 2))])

        np.savetxt('image_mean.txt', images_mean, fmt='%.8f')
        np.savetxt('image_std.txt', images_std, fmt='%.8f')
        np.savetxt('label_mean.txt', labels_mean, fmt='%.8f')
        np.savetxt('label_std.txt', labels_std, fmt='%.8f')

    return images_mean, images_std, labels_mean, labels_std

if __name__ == '__main__':
    from config import args
    import constants
    
    dim = args.input_height
    scale = dim/constants.default_width    
    print "Generating normalization (dim={}, scale={})...".format(dim, scale)
    normalization_values(scale=scale, root_dir='B_old/', csv_file='B_old/train_data.csv', cache=False)