# Deploy model pipline
from __future__ import print_function, division

import tqdm
import sys
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
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import LabelExtrator

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == "__main__":

    # (1) Loading model
    model = facenet_resnet(nb_class=71, embdding=256, is_train=False)
    model.load_weights("models/model.h5")

    pred_labels = []
    gt_labels = []

    if not os.path.exists("results"):
        os.mkdir("results")

    # (2) Image processing and cropping
    for image_path in tqdm.tqdm(sorted(paths.list_images("test_images"))):
        _, name = os.path.split(image_path)

        raw = cv.imread(image_path)  # BGR
        img = Image.open(image_path).convert('RGB')

        boxes, facial5points = detect_faces(img)

        # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
        if len(facial5points) != 1:
            pred_labels.append(np.random.randint(2))
            continue

        facial5points = np.reshape(facial5points[0], (2, 5))
        box = boxes[0].astype(int)

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

        # (4) Predict gender and other attributes
        # gender_class = np.argmax(data_results[0]) # softmax
        gender_class = 1 if data_results[0] > 0.5 else 0 # sigmoid
        pred_labels.append(gender_class)
    
    # Calculate F1 score
    with open("labels.txt", 'r') as f:
        for line in f.readlines():
            gt_labels.append(int(line.strip().split()[1]))
    f1_score = f1_score(gt_labels, pred_labels, average='weighted')
    print("F1 score: {}".format(f1_score))

    # Correlation with Face++
    for json_path in tqdm.tqdm(sorted(paths.list_files("./test_images_json"))):
        face = LabelExtrator.FaceLabels(json_path).getFace(0)
        gender = np.argmax(face.Gender)
        gt_labels.append(gender)

    corr = matthews_corrcoef(gt_labels, pred_labels)
    print("Correction: {}".format(corr))