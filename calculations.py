"""Calculations needed for neural networks"""

def conv_dim(width, height, padding, filter_size, stride):
    # (n + 2p - f)/ s + 1
    output_width = (width + 2 * padding - filter_size) / stride + 1
    output_height = (height + 2 * padding - filter_size) / stride + 1
    return output_width, output_height


def max_pool_dim(width, height, v):
    output_width = width / v
    output_width = height / v
    return output_width, output_width