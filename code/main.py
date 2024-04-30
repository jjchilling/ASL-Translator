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
        '--train',
        default=False,
        help='Whether or not we train the model.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def classify_image(datasets, model):
    test = datasets.get_data("../data/test", True, True, True)
    count = 0
    predictions = []

    for batch in test:
        if (count==29):
            break
        for i, image in enumerate(batch[0]):
            correct_class_idx = batch[1][i]
            #probabilities = model.vgg16(np.array([image])).numpy()[0]
            output = model.call(np.array([image]))
            probabilities = output.numpy()[0]
            predict_class_idx = np.argmax(probabilities)
            predictions.append(predict_class_idx)
            prediction_label = datasets.idx_to_class[predict_class_idx]
            print("Predicted label:", prediction_label)

            # This undoes vgg processing from stencil
            mean = [103.939, 116.779, 123.68]
            image[..., 0] += mean[0]
            image[..., 1] += mean[1]
            image[..., 2] += mean[2]
            image = image[:, :, ::-1]
            image = image / 255.
            image = np.clip(image, 0., 1.)
            
            #shows the image so you can compare it to the predicted label in the terminal
            plt.imshow(image)
            plt.show()

        count += 1
    # prediction = np.argmax(predictions)
    # print("prediction: ", prediction)


def main():

    datasets = Datasets(ARGS.data, '3')

    # just for training the model
    if ARGS.train == 'True':
        
        time_now = datetime.now()
        timestamp = time_now.strftime("%m%d%y-%H%M%S")
        init_epoch = 0
        os.chdir(sys.path[0])
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model = VGGModel()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
        os.makedirs(checkpoint_path)
        model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

        train(model, datasets, checkpoint_path, logs_path, init_epoch)


############################################################
    else :
        """ Main function. """
        # loading model
        model = VGGModel()
        model(tf.keras.Input(shape=(224, 224, 3)))
        model.vgg16.load_weights('vgg16_imagenet.h5', by_name=True)
        # model.head.load_weights('checkpoints/vgg_model/042324-233248/vgg.weights.e026-acc0.9286.h5', by_name=False)
        model.head.load_weights('checkpoints/vgg_model/043024-144031//vgg.weights.e012-acc0.9655.h5', by_name=False)
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss_fn,
            metrics=["sparse_categorical_accuracy"])
        classify_image(datasets, model)
            

# Make arguments global
ARGS = parse_args()
# live()
main()
