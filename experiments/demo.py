# coding: utf-8
import tensorflow as tf
import shutil
import random
import matplotlib.pyplot as plt
import time
from keras.models import load_model
import dlib
import numpy as np
import cv2
import os
from face_alignment import Face_aligned
from utils import mk_dir
import imutils


# def copyfile(file_dir, tar_dir, n=20):
#     '''
#     Randomly select n images from the data set for testing
#     '''
#     if os.path.exists(tar_dir):
#         shutil.rmtree(tar_dir)
#     time.sleep(1e-10)
#     mk_dir(tar_dir)

#     (filepath, filename) = os.path.split(file_dir)
#     if filename == "UTKFace":
#         path_dir = [t_dir for t_dir in os.listdir(
#             file_dir) if t_dir.split("_")[2] == '2']
#     else:
#         path_dir = os.listdir(file_dir)
#     sample = random.sample(path_dir, n)

#     for name in sample:
#         shutil.copyfile(os.path.join(file_dir, name),
#                         os.path.join(tar_dir, name))


if __name__ == '__main__':
    # FaceNet
    IMAGE_SIZE = 160
    PADDING = 0.2
    WEIGHTS = "../models/model.h5"
    base_path = "test"
    gender_list = ['W', 'M']
    mk_dir("results")

    tf.logging.set_verbosity(tf.logging.ERROR)
    faceAlign = Face_aligned(keras_model=WEIGHTS)

    for _, _, imgs in os.walk(base_path):
        '''
        Only support one face in the image
        '''
        for idx, im in enumerate(imgs):
            img_path = os.path.join(base_path, im)

            # Extract image suffix name
            (_, name) = os.path.split(img_path)
            image = cv2.imread(img_path)

            try:
                image.shape
            except:
                continue

            # in order to reduce inference time
            image = imutils.resize(image, width=256)
            s = time.time()
            img_data, boxes = faceAlign.face_align(
                image, padding=PADDING, size=IMAGE_SIZE)
            d = time.time() - s
            print("[INFO] Completion time used: {}".format(d))
            
            if img_data is not None and boxes is not None:
                s = time.time()
                results = faceAlign.predict_image(img_data)
                d = time.time() - s
                print("[INFO] Inference time used: {}".format(d))

                predicted_gender = gender_list[np.argmax(results[0])]
                ages = np.arange(0, 71).reshape(71, 1)
                predicted_age = int(
                    np.round(results[1].dot(ages).flatten()[0], 1))
            else:
                print("[INFO] Big brother frontal, single person...")
                continue

            # plot visualization result
            '''img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            plt.imshow(img_data_rgb)
            plt.title("{}, {}".format(predicted_gender, predicted_age))
            plt.axis('off')
            plt.show()'''

            if predicted_gender == 'W':
                cv2.rectangle(
                    image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)
            else:
                cv2.rectangle(
                    image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 0, 255), 2)

            cv2.putText(image, 'age: {}'.format(predicted_age), (boxes[0], boxes[1]+30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            # cv2.imwrite("results/{}.png".format(name), image)
            cv.imshow("image", image)
            cv.waitKey(-1)
