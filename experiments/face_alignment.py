from keras.models import load_model
import imutils
import timeit
import time
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
import detect_face

import sys
sys.path.append("../")
from preprocessor import preprocess_input
from models.facenet import facenet_resnet


def resize_image(image, size):
    '''
    Rectangular complements are square
    '''
    top, bottom, left, right = 0, 0, 0, 0
    h, w, _ = image.shape
    longest_edge = max(h, w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    constant = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    return cv2.resize(constant, (size, size))


class Face_aligned:

    """Align the face area according to the five feature-point and crop the image"""

    def __init__(self, factor=0.709, threshold=[0.6, 0.7, 0.7], minsize=20, embdding=256, keras_model=None):
        if isinstance(threshold, list) and len(threshold) == 3:
            self.threshold = threshold
            self.factor = factor
            self.minsize = minsize
            # self.predictor = predictor
            self.embdding = embdding

            # initial mtcnn network
            with tf.Graph().as_default():
                sess = tf.Session()
                with sess.as_default():
                    self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(
                        sess, None)
            try:
                self.model = load_model(keras_model)
            except:
                try:
                    self.model = facenet_resnet(
                        nb_class=71, embdding=self.embdding, is_train=False)
                    self.model.load_weights(keras_model)
                except:
                    raise ValueError("weight's path having not loading...")

    def face_align(self, img, padding, size):
        """
        Alignment algorithm implementation details
        img: BGR image
        padding: image padding
        size: required image size 
        """
        try:
            img_h, img_w, _ = img.shape
        except:
            raise ValueError("reading image error...")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = detect_face.detect_face(
            img_rgb, self.minsize, self.pnet, self.rnet, self.onet, threshold=self.threshold, factor=self.factor)
        boxes = self.__proprecess(boxes)

        if boxes is None:
            return None, None

        x1, y1, x2, y2, w, h = boxes[0], boxes[1], boxes[2], boxes[3], \
            boxes[2] - boxes[0], boxes[3] - boxes[1]
        xw1 = max(int(x1 - padding * w), 0)
        yw1 = max(int(y1 - padding * h), 0)
        xw2 = min(int(x2 + padding * w), img_w - 1)
        yw2 = min(int(y2 + padding * h), img_h - 1)

        image = resize_image(img[yw1: yw2 + 1, xw1: xw2 + 1, :], size=size)

        return image, boxes

    def __proprecess(self, boxes):
        """Preprocessing the returned boxes and points"""
        # assert len(boxes) == 1, "only one face is vaild..."
        if len(boxes) == 1:
            boxes = np.array(list(map(int, boxes[:, 0:4][0])))
            return boxes
        else:
            return None

    def predict_image(self, image):
        """Keras predict gender and age"""
        img_data = image[np.newaxis, :]
        img_data = preprocess_input(img_data)
        results = self.model.predict(img_data)

        return results
