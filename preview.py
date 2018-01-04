import numpy as np
import cv2
import constants


def load_preview(images, outputs, labels_std,
                 labels_mean, images_std, images_mean):
    """This function logs a preview image to tensorboard"""
    image = images.data[0].cpu().numpy()
    output = outputs.data[0].cpu().numpy()
    image = np.rollaxis(image, 0, 3)

    output = output.reshape(-1, 2)
    output = output * labels_std + labels_mean

    image = image * images_std + images_mean
    image = image.squeeze()
    image = image / 255
    image = image.copy()

    if constants.grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for coordinates in output:
        x = coordinates[0]
        y = coordinates[1]
        circle_size = 2
        cv2.circle(image, (int(x), int(y)), circle_size, (0, 0, 255), -1)

    return image
