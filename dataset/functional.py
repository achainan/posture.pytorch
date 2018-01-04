import torch
import numpy as np
import scipy.ndimage as ndi
import cv2

def normalize_image(image, mean, std):
    image = (image - mean) / std
    image[image == np.inf] = 0
    image[image == -np.inf] = 0
    image[np.isnan(image)] = 0
    return image
        
def resize(image, size):
    # image = scipy.misc.imresize(image, self.size)
    image = cv2.resize(image, None, fx=size, fy=size)
    return image

def image_to_tensor(image):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image.float()
        
def square(image):
    height = image.shape[0]
    width = image.shape[1]
    x = 0
    y = 0
    shape = (height, width, 3)
    if width > height:
        y = (height - width)/2
        shape = (width, width, 3)
    elif height > width:
        x = (height - width)/2
        shape = (height, height, 3)
        
    square = np.zeros((shape), np.float32)
    square[y:y+height, x:x+width] = image

    return square, x, y
    
def shift(image, offset):
    channel_axis = 2
    cval = 0.
    tx = np.random.randint(-1 * offset, high=offset)
    ty = 0  # Currently only horizontal shift
    translation_matrix = np.array([[1, 0, ty],
                                   [0, 1, tx],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    image = apply_transform(image, transform_matrix,
                            channel_axis, 'nearest', cval)
    return image, tx, ty
    
# Taken from keras:
# https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py

def apply_transform(x, transform_matrix, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Apply the image transformation specified by a matrix."""

    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x