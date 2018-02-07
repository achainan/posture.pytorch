import torch
import numpy as np
import scipy.ndimage as ndi
import cv2
from torch.autograd import Variable
import random
import math
from PIL import Image
  
def denormalize_image_tensor(tensor_mean, tensor_std, images, cuda): 
    tensor_std = Variable(tensor_std.float().squeeze())
    tensor_mean = Variable(tensor_mean.float().squeeze())
    if cuda:
        tensor_std = tensor_std.cuda()
        tensor_mean = tensor_mean.cuda()           
    images = images.permute(0, 2, 3, 1)
    images = images * tensor_std + tensor_mean
    images = images.permute(0, 3, 1, 2)        
    return images

def full(img, shape):
    img = np.array(img, copy=True, dtype=np.float32)
    result = np.full(shape, 255, np.float32)
    w = img.shape[1]
    h = img.shape[0]
    x = (shape[1] - w) / 2
    y = (shape[0] - h) / 2
    result[y:y+h, x:w+x] = img
    return result
        
def swap_colors(input, random):
    output = np.zeros((input.shape), np.float32)
    
    red, green, blue = input[:,:,0].copy(), input[:,:,1].copy(), input[:,:,2].copy()
    
    prob = 1.0/6.0
    
    if random < prob:
        output[:,:,0], output[:,:,1], output[:,:,2] = red, green, blue
    elif random < prob*2:
        output[:,:,0], output[:,:,1], output[:,:,2] = red, blue, green
    elif random < prob*3:
        output[:,:,0], output[:,:,1], output[:,:,2] = green, red, blue
    elif random < prob*4:
        output[:,:,0], output[:,:,1], output[:,:,2] = green, blue, red
    elif random < prob*5:
        output[:,:,0], output[:,:,1], output[:,:,2] = blue, green, red
    else:
        output[:,:,0], output[:,:,1], output[:,:,2] = blue, red, green

    return output
    
def normalize_image(image, mean, std):
    image = (image - mean) / std
    image[image == np.inf] = 0
    image[image == -np.inf] = 0
    image[np.isnan(image)] = 0
    return image
        
def deprecated_resize(image, size):
    # image = scipy.misc.imresize(image, self.size)
    image = cv2.resize(image, None, fx=size, fy=size)
    return image
        
def square(image):
    height = image.shape[0]
    width = image.shape[1]
    x = 0
    y = 0
    shape = (height, width, 3)
    if height == width:
        return image, x, y 
    elif width > height:
        y = (height - width)/2
        shape = (width, width, 3)
    elif height > width:
        x = (height - width)/2
        shape = (height, height, 3)
        
    square = np.zeros((shape), np.float32)
    square[y:y+height, x:x+width] = image

    return square, x, y
    
# def shift(image, offset):
#     channel_axis = 2
#     cval = 0.
#     tx = random.randint(int(-1 * offset), int(offset))
#     ty = 0  # Currently only horizontal shift
#     translation_matrix = np.array([[1, 0, ty],
#                                    [0, 1, tx],
#                                    [0, 0, 1]])
#
#     transform_matrix = translation_matrix  # no need to do offset
#     image = apply_transform(image, transform_matrix, channel_axis, 'nearest', cval)
#     return image, tx, ty

def shift(image, offset):
    tx = random.randint(int(-1 * offset), int(offset))
    ty = 0  # Currently only horizontal shift
    
    new_center = (-tx, -ty)
    center = (0,0)
    nx,ny = x,y = center
    (nx,ny) = new_center
    cosine = 1.0
    sine = 0.0
    a = cosine
    b = sine
    c = x-nx*a-ny*b
    d = -sine
    e = cosine
    f = y-nx*d-ny*e
    image = image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.BICUBIC)
            
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