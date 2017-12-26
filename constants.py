"""While in development instead of args using a constants file"""

num_epochs = 2000
train_batch_size = 114
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
        images_mean = 218.10636902
        images_std = 51.42391586
        labels_mean = 141.24706
        labels_std = 59.869335
    else:
        images_mean = [225.21047974, 215.81893921, 211.25111389]
        images_std = [49.95010757, 54.02108765, 53.14689255]
        labels_mean = 141.24706
        labels_std = 59.869335

    return images_mean, images_std, labels_mean, labels_std
