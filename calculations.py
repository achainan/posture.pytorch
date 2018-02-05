"""Calculations needed for neural networks"""


def conv_dim(width, height, padding, filter_size, stride):
    # (n + 2p - f)/ s + 1
    output_width = (width + 2.0 * padding - filter_size) / stride * 1.0 + 1
    output_height = (height + 2.0 * padding - filter_size) / stride * 1.0 + 1
    return int(output_width), int(output_height)


def max_pool_dim(width, height, v):
    output_width = width / v * 1.0
    output_height = height / v * 1.0
    return int(output_width), int(output_height)
