"""While in development instead of args using a constants file"""

num_epochs = 2000
train_batch_size = 76
val_batch_size = 12
learning_rate = 0.001
num_workers = 10
scale = .4
default_width = 670
default_height = 760
scaled_width = int(scale * default_width)
scaled_height = int(scale * default_height)
save_interval = 20
print_freq = 10


def normalization_values(grayscale=True):
    if grayscale:
        images_mean = 218.12825012
        images_std = 51.27311325
        labels_mean = 147.459
        labels_std = 57.4442
    else:
        images_mean = [225.08126831, 215.9152832, 211.28643799]
        images_std = [49.91968536, 53.75146866, 53.13668823]
        labels_mean = 147.459
        labels_std = 57.4442

    return images_mean, images_std, labels_mean, labels_std
