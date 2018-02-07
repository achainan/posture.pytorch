import numpy as np
import cv2
import constants

def load_preview(image, output, circle_size=10, grayscale=False):
    """This function logs a preview image to tensorboard"""
    image = image.squeeze()
    image = image.copy()

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return annotate(image, output, circle_size)


def annotate(image, output, circle_size=10, color=(255, 0, 0)):
    for coordinates in output:
        x = coordinates[0]
        y = coordinates[1]
        cv2.circle(image, (int(round(x)), int(round(y))), circle_size, color, -1)
    return image
