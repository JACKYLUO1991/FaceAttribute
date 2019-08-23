from scipy.io import loadmat
from datetime import datetime
import os
import dlib
import cv2
import numpy as np

from collections import Counter, OrderedDict


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass


def count_multiclass_num(class_list):
    """Count the number of different types of elements"""
    assert isinstance(class_list, list)
    dict_class = dict(Counter(class_list))
    multiclass_num = list(OrderedDict(
        sorted(dict_class.items())).values())

    return multiclass_num
