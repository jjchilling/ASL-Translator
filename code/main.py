"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import VGGModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np
from live import live
import skimage.segmentation as seg
import copy
from sklearn.utils import check_random_state

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = 0

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


def classify_image():
    datasets = Datasets(ARGS.data, ARGS.task)
    test = datasets.get_data("../data/test/", True, True, True)
    count = 0
    predictions = []
    for batch in test:
        if (count==25):
            break
        for i, image in enumerate(batch[0]):
            correct_class_idx = batch[1][i]
            probabilities = model(np.array([image])).numpy()[0]
            predict_class_idx = np.argmax(probabilities)
            predictions.append(predict_class_idx)
        count += 1
    prediction = np.argmax(predictions)
    print("prediction: ", prediction)


def main():
    """ Main function. """

    # loading model
    model = VGGModel()
    model(tf.keras.Input(shape=(224, 224, 3)))
    model.vgg16.load_weights('vgg16_imagenet.h5', by_name=True)
    model.head.load_weights('checkpoints/vgg_model/042324-233248/vgg.weights.e026-acc0.9286.h5', by_name=False)
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])


    classify_image()
    # test = datasets.get_data("../data/test/", True, True, True)
    # count = 0
    # predictions = []
    # for batch in test:
    #     if (count==25):
    #         break
    #     for i, image in enumerate(batch[0]):
    #         correct_class_idx = batch[1][i]
    #         probabilities = model(np.array([image])).numpy()[0]
    #         predict_class_idx = np.argmax(probabilities)
    #         predictions.append(predict_class_idx)
    #     count += 1
    # prediction = np.argmax(predictions)
    # print("prediction: ", prediction)
            

# Make arguments global
ARGS = parse_args()
# live()
main()
