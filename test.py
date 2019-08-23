from __future__ import print_function, division

import tqdm
import os
from PIL import Image
import cv2 as cv
import numpy as np
import time

from preprocessor import preprocess_input
from mtcnn.detector import detect_faces
from align_faces import warp_and_crop_face, get_reference_facial_points
from models.facenet import facenet_resnet
from imutils import paths
import imutils

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == "__main__":

    # (1) Loading model
    model = facenet_resnet(nb_class=71, embdding=256, is_train=False)
    model.load_weights("models/model.h5")
    # model.summary()

    gCategory = ['M', 'W']

    if not os.path.exists("results"):
        os.mkdir("results")

    # (2) Image processing and cropping
    for image_path in tqdm.tqdm(sorted(paths.list_images("./demo"))): # Based on the path of the data
        _, name = os.path.split(image_path)

        raw = cv.imread(image_path)  # BGR
        raw = imutils.resize(raw, width=256)
        # img = Image.fromarray(raw[:, :, ::-1]) # RGB
        
        t0 = time.time()
        boxes, facial5points = detect_faces(raw)
        
        # whether show face keypoints or not
        # for p in facial5points:
        #     for i in range(5):
        #         cv.circle(raw, (int(p[i]), int(p[i + 5])), 4, (255, 0, 255), -1)
        # if len(facial5points) != 1:
        #     continue
        # facial5points = np.reshape(facial5points[0], (2, 5))
        # box = boxes[0].astype(int)
        if boxes is not None:
            for i in range(boxes.shape[0]):
                box = boxes[i].astype(np.int)
                if facial5points is not None:
                     facial5points = facial5points[i].astype(np.int)
                     for l in range(facial5points.shape[0]):
                        cv.circle(raw, (facial5points[l][0], facial5points[l][1]), 4, (255, 0, 255), -1)
        else:
            print("No face detection or multi faces...")
            continue
        
        # setting face alignment parameters
        crop_size = 160
        default_square = True
        inner_padding_factor = 0.1
        outer_padding = 0
        output_size = 160
        
        reference_5pts = get_reference_facial_points(
            (output_size, output_size), inner_padding_factor, (outer_padding, outer_padding), default_square)
        
        dst_img = warp_and_crop_face(
            raw, facial5points, reference_pts=reference_5pts, crop_size=(
                crop_size, crop_size)
        )
        
        # (3) Convert image data to keras format
        img_data = dst_img[np.newaxis, :]
        img_data = preprocess_input(img_data)
        data_results = model.predict(img_data)
        t1 = time.time() - t0
        print("Time used: {} ms".format(t1 * 1000))
        
        # (4) Predict gender and other attributes
        predicted_gender = gCategory[np.argmax(data_results[0])]
        ages = np.arange(0, 71).reshape(71, 1)
        predicted_age = int(
            np.round(data_results[1].dot(ages).flatten()[0], 1))

        # (5) Draw images
        if predicted_gender == 'W':
            cv.rectangle(raw, (box[0], box[1]),
                         (box[2], box[3]), (0, 0, 255), 2)
        else:
            cv.rectangle(raw, (box[0], box[1]),
                         (box[2], box[3]), (0, 255, 255), 2)
        
        cv.putText(raw, 'age: {}'.format(predicted_age), (box[0], box[1]+30),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv.imwrite("results/{}".format(name), raw)
