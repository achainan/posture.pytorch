import numpy as np
import cv2
import constants

def load_preview(images, outputs, labels_std=1, labels_mean=0, images_std=1, images_mean=0):
    """This function logs a preview image to tensorboard"""
    image = images.data[0].cpu().numpy()
    image = np.rollaxis(image, 0, 3)
    image = image * images_std + images_mean
    image = image.squeeze()
    image = image / 255
    image = image.copy()

    output = outputs.data[0].cpu().numpy()
    output = output.reshape(-1, 2)
    output = output * labels_std + labels_mean

    if constants.grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return annotate(image, output)


def annotate(image, output):
    for coordinates in output:
        x = coordinates[0]
        y = coordinates[1]
        circle_size = int(constants.scale*10)
        cv2.circle(image, (int(x), int(y)), circle_size, (0, 0, 255), -1)
    return image
