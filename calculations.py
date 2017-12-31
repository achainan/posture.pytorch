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


def pad_to_square(width, height):
    padding_left = padding_right = padding_top = padding_bottom = 0
    padding_vertical = padding_horizontal = 0
    if width > height:
        padding_vertical = width - height
        padding_horizontal = 0
        padding_left = padding_right = 0
        padding_top = padding_vertical / 2 + padding_vertical % 2
        padding_bottom = padding_vertical / 2
    else:
        padding_vertical = 0
        padding_horizontal = height - width
        padding_left = padding_horizontal / 2 + padding_horizontal % 2
        padding_right = padding_horizontal / 2
        padding_top = padding_bottom = 0

    return padding_left, padding_right, padding_top, padding_bottom
