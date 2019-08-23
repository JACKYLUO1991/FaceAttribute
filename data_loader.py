from __future__ import print_function, division

import sys
from scipy.io import loadmat
import numpy as np
from utils import calc_age
import cv2
from pathlib import Path
import os
import pandas as pd
import LabelExtrator as tools
from tqdm import tqdm

import random
random.seed(2018)

from utils import count_multiclass_num


class DataManager(object):
    """
    Perform dataset management operations
    """

    def __init__(self, dataset_name='megaage_asian'):
        self.dataset_name = dataset_name

        if self.dataset_name == 'imdb':
            self.dataset_path = 'data/imdb_crop/imdb.mat'
        elif self.dataset_name == 'wiki':
            self.dataset_path = 'data/wiki_crop/wiki.mat'
        elif self.dataset_name == 'utkface':
            self.dataset_path = 'data/UTKFace/'
        elif self.dataset_name == 'adience':
            self.dataset_path = 'data/adience.csv'
        if self.dataset_name == 'megaage_asian':
            self.dataset_path = 'data/megaage_asian_train/json_files.txt'
        else:
            raise Exception('Invalid dataset')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        if self.dataset_name == 'wiki':
            ground_truth_data = self._load_wiki()
        if self.dataset_name == 'utkface':
            ground_truth_data = self._load_utkface()
        if self.dataset_name == 'adience':
            ground_truth_data = self._load_adience()
        if self.dataset_name == 'megaage_asian':
            ground_truth_data = self._load_megaage_asian()
        return ground_truth_data

    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        # 0 for Female and 1 for Male, NaN if unknown
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        # date of birth
        dob = dataset['imdb']['dob'][0, 0][0]
        # year when the photo was taken
        photo_taken = dataset['imdb']['photo_taken'][0, 0][0]

        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        age_classes = np.array([calc_age(photo_taken[i], dob[i])
                                for i in range(len(dob))])

        # 0 <= age_classes <= 100
        valid_age_range = np.isin(age_classes, [x for x in range(101)])

        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        mask = np.logical_and(mask, valid_age_range)

        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        age_classes = age_classes[mask].tolist()

        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)

        return dict(zip(image_names, zip(gender_classes, age_classes)))

    def _load_wiki(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['wiki']['full_path'][0, 0][0]
        # 0 for Female and 1 for Male, NaN if unknown
        gender_classes = dataset['wiki']['gender'][0, 0][0]
        face_score = dataset['wiki']['face_score'][0, 0][0]
        second_face_score = dataset['wiki']['second_face_score'][0, 0][0]
        # date of birth
        dob = dataset['wiki']['dob'][0, 0][0]
        # year when the photo was taken
        photo_taken = dataset['wiki']['photo_taken'][0, 0][0]

        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        age_classes = np.array([calc_age(photo_taken[i], dob[i])
                                for i in range(len(dob))])

        # 0 <= age_classes <= 100
        valid_age_range = np.isin(age_classes, [x for x in range(101)])

        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        mask = np.logical_and(mask, valid_age_range)

        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        age_classes = age_classes[mask].tolist()

        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)

        return dict(zip(image_names, zip(gender_classes, age_classes)))

    def _load_utkface(self):
        image_names = []
        gender_classes = []
        age_classes = []
        image_dir = Path(self.dataset_path)
        for i, image_path in enumerate(image_dir.glob("*.jpg")):
            image_name = image_path.name
            age, gender = image_name.split("_")[:2]
            try:
                if int(gender) == 0:
                    gender = 1
                elif int(gender) == 1:
                    gender = 0
                else:
                    continue
            except:
                continue
            # Swap the order of the gender
            gender_classes.append(gender)
            age_classes.append(min(int(age), 100))
            image_names.append(image_name)

        return dict(zip(image_names, zip(gender_classes, age_classes)))

    def _load_adience(self):
        data = pd.read_csv(os.path.join(self.dataset_path))
        image_names = data['full_path'].values
        age_classes = data['age'].values.astype('uint8')
        gender_classes = data['gender'].values.astype('uint8')

        return dict(zip(image_names, zip(gender_classes, age_classes)))

    def _load_megaage_asian(self, base_path="data/megaage_asian_train"):
        """
        Load megaage asian dataset
        """
        image_names = []
        gender_classes = []
        age_classes = []
        emotion_classes = []

        with open(self.dataset_path) as f:
            lines = f.readlines()

            # for line in tqdm(lines):
            for line in lines:
                line = line.strip()
                image_path = line.replace(".json", ".jpg")

                try:
                    # Call face++ interface
                    face = tools.FaceLabels(
                        os.path.join(base_path, line)).getFace(0)
                except:
                    continue

                gender = np.argmax(face.Gender)
                emotion = np.argmax(face.Emotion)
                # range from 0 to 70
                age = 70 if face.Age > 70 else face.Age
                age = np.array(age, dtype=np.uint8)

                image_names.append(image_path)
                gender_classes.append(gender)
                age_classes.append(age)
                emotion_classes.append(emotion)

        return dict(zip(image_names, zip(gender_classes, age_classes, emotion_classes)))

def split_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle:
        random.shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys
