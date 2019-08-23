import numpy as np
from scipy.misc import imread, imresize


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def _imread(image_name):
        return imread(image_name)


def _imresize(image_array, size):
        return imresize(image_array, size)
