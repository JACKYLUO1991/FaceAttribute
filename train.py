from __future__ import print_function, division

import argparse
import pandas as pd
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.models import Model
from data_generator import ImageGenerator
from data_loader import DataManager, split_data
from models.facenet import facenet_resnet
from utils import mk_dir
from metrics import *
import tensorflow as tf
import keras.backend as K
# from SGDR import SGDRScheduler
# from CLR import CyclicLR
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database")
    parser.add_argument("--dataset_name", '-d', type=str, required=True,
                        help="name of dataset")
    parser.add_argument("--embdding", "-e", type=int, required=True,
                        help="embdding of the model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--image_size", type=int, default=160,
                        help="image size")
    parser.add_argument("--class_num", type=int, default=71,
                        help="age class number")
    parser.add_argument("--lr", type=float, default=2.5e-3,
                        help="initial learning rate")
    args = parser.parse_args()

    return args

# [注]：论文中用的multi step方法，这里使用更加实用的PolyDecay方法！
class PolyDecay:

    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs

    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0 * epoch / self.n_epochs, self.power)


def main():
    # Set GPU memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = get_args()
    IMG_SIZE = args.image_size
    input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    dataset_name = args.dataset_name
    n_age_bins = args.class_num
    embdding = args.embdding
    lr = args.lr

    logging.debug("[INFO] Loading data...")

    data_loader = DataManager(dataset_name)
    ground_truth_data = data_loader.get_data()
    train_keys, val_keys = split_data(
        ground_truth_data, validation_split=validation_split, do_shuffle=True)

    print("Samples: Training - {}, Validation - {}".format(len(train_keys), len(val_keys)))
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                     input_shape[:2],
                                     train_keys, val_keys,
                                     path_prefix=input_path,
                                     vertical_flip_probability=0,
                                     eraser_probability=0,
                                     bins=n_age_bins)

    model = facenet_resnet(nb_class=n_age_bins, embdding=embdding, is_train=True,
                           weights="./models/facenet_keras_weights.h5")
    model.compile(
        optimizer=optimizers.SGD(
            lr=lr, momentum=0.9, decay=5e-4, nesterov=False),
        loss={'pred_g': focal_loss(alpha=.4, gamma=2),
              'pred_a': mae, "pred_e": "categorical_crossentropy"},
        loss_weights={'pred_g': 0.2, 'pred_a': 1, 'pred_e': 0.4}, 
        metrics={'pred_g': 'accuracy',
                 'pred_a': mae, 'pred_e': 'accuracy'})

    logging.debug("[INFO] Saving model...")

    mk_dir("checkpoints")

    callbacks = [
        CSVLogger(os.path.join('checkpoints', 'train.csv'), append=False),
        ModelCheckpoint(
            os.path.join(
                'checkpoints', 'weights.{epoch:02d}-{val_pred_g_acc:.3f}-{val_pred_a_mae:.3f}-{val_pred_e_acc:.3f}.h5'),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min"
        ),
        # Use Stochastic Gradient Descent with Restart
        # https://github.com/emrul/Learning-Rate
        # Based on paper SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
        # SGDRScheduler(min_lr=lr*((0.1)**3), max_lr=lr, steps_per_epoch=np.ceil(len(train_keys) /
        #                                                                        batch_size), lr_decay=0.9, cycle_length=5, mult_factor=1.5)
        
        # Use Cyclical Learning Rate
        # CyclicLR(mode='triangular', step_size=np.ceil(
        #     len(train_keys)/batch_size), base_lr=lr*((0.1)**3), max_lr=lr)
        LearningRateScheduler(PolyDecay(lr, 0.9, nb_epochs).scheduler)
    ]

    logging.debug("[INFO] Running training...")

    history = model.fit_generator(
        image_generator.flow(mode='train'),
        steps_per_epoch=np.ceil(len(train_keys)/batch_size),
        epochs=nb_epochs,
        callbacks=callbacks,
        validation_data=image_generator.flow('val'),
        validation_steps=np.ceil(len(val_keys)/batch_size)
    )

    logging.debug("[INFO] Saving weights...")

    K.clear_session()


if __name__ == '__main__':
    main()
