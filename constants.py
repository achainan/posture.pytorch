"""While in development instead of args using a constants file"""

train_batch_size = 3  # Since we have so little data currently
val_batch_size = 1  # Since we have so little data currently
learning_rate = 0.001
input_height = 32.0
scale = input_height/760.0
default_width = 670
default_height = 760
scaled_width = int(scale * default_width)
scaled_height = int(scale * default_height)
save_interval = 20
print_freq = 1
display_freq = 10  # Save preview to tensorboard ever X epochs
grayscale = False
