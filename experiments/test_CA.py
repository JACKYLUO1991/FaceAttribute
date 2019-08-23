# measure CA(3) and CA(5), CA(7)
# Quantifying Facial Age by Posterior of Age Comparisons

import numpy as np
import os
import cv2
import dlib
import imutils
import csv
import logging
import time
from face_alignment import Face_aligned
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == "__main__":
    data_path = "../data/megaage_asian_test"
    age_file = np.loadtxt("../data/megaage_asian_list/test_age.txt")
    img_name_file = np.genfromtxt(
        "../data/megaage_asian_list/test_name.txt", dtype="str")

    image_array = []
    age_array = []

    for i in range(len(img_name_file)):
        img_path = os.path.join(data_path, img_name_file[i])
        img = cv2.imread(img_path)
        try:
            img.shape
        except:
            continue

        image_array.append(str(img_path))
        age_array.append(int(age_file[i]))

    # FaceNet
    IMAGE_SIZE = 160
    PADDING = 0.2
    WEIGHTS = "../models/model.h5"

    # loading model
    faceAlign = Face_aligned(keras_model=WEIGHTS)

    CA3 = 0
    CA5 = 0
    CA7 = 0
    pred = []

    for i in tqdm(range(len(image_array))):
        # logging.debug("loading image {}".format(i))

        img = cv2.imread(image_array[i])
        age = age_array[i]

        image = imutils.resize(img, width=128)
        img_data, boxes = faceAlign.face_align(
            image, padding=PADDING, size=IMAGE_SIZE)

        if img_data is not None and boxes is not None:
            results = faceAlign.predict_image(img_data)
            ages = np.arange(0, 71).reshape(71, 1)
            age_pred = results[1].dot(ages).flatten()[0]
        else:
            continue

        error = np.absolute(age - age_pred)

        if error <= 3:
            CA3 += 1
        if error < 5:
            CA5 += 1
        if error < 7:
            CA7 += 1
        pred.append(error)

    CA3 /= len(pred)
    CA5 /= len(pred)
    CA7 /= len(pred)

    print('CA3: ', CA3, 'CA5: ', CA5, 'CA7: ', CA7)

    # write into csv file
    # 0.6228093120585927,0.8153282762228616,0.9170808265759874
    dataframe = pd.DataFrame({'CA3': [CA3], 'CA5': [CA5], 'CA7': [CA7]})
    dataframe.to_csv("CA.csv", index=False, sep=",")



