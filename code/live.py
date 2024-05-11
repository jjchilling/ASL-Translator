#!/usr/bin/env python
#coding: utf8



"""
Code originally by Brian R. Pauw and David Mannicke.
Modified by James Tompkin for Brown CSCI1430.

Initial Python coding and refactoring:
    Brian R. Pauw
With input from:
    Samuel Tardif

Windows compatibility resolution: 
    David Mannicke
    Chris Garvey

Windows compiled version:
    Joachim Kohlbrecher
"""

"""
Overview
========
This program uses OpenCV to capture images from the camera, Fourier transform
them and show the Fourier transformed image alongside the original on the screen.

$ ./liveFFT2.py

Required: A Python 3.x installation (tested on 3.7.9),
with: 
    - OpenCV (for camera reading)
    - numpy, matplotlib, scipy, argparse
"""

__author__ = "Brian R. Pauw, David Mannicke; modified for Brown CSCI 1430 by James Tompkin"
__contact__ = "brian@stack.nl; james_tompkin@brown.edu"
__license__ = "GPLv3+"
__date__ = "2014/01/25; modifications 2017--2019"
__status__ = "v2.1"

import cv2 # opencv-based functions
import time
import math
import numpy as np
from scipy import stats
from skimage import io
from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray
from preprocess import Datasets
from models import VGGModel
import os
import tensorflow as tf
from matplotlib import pyplot as plt



class live():
    """
    This function shows the live Fourier transform of a continuous stream of 
    images captured from an attached camera.

    """

    wn = "FD"
    use_camera = True
    im = 0
    imJack = 0
    phaseOffset = 0
    rollOffset = 0
    # Variable for animating basis reconstruction
    frequencyCutoffDist = 1
    frequencyCutoffDirection = 1
    # Variables for animated basis demo
    magnitude = 2
    orientation = 0

    def __init__(self, **kwargs):

        model = VGGModel()
        model(tf.keras.Input(shape=(224, 224, 3)))
        model.vgg16.load_weights('vgg16_imagenet.h5', by_name=True)
        model.head.load_weights('checkpoints/vgg_model/043024-144031//vgg.weights.e012-acc0.9655.h5', by_name=False)
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss_fn,
            metrics=["sparse_categorical_accuracy"])

        def classify_image(model, labels):
            datasets = Datasets('..'+os.sep+'data'+os.sep, '3')
            test = datasets.get_data(os.getcwd()+os.sep+'frame'+os.sep+'test', True, True, True)
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
                    # print("Predicted label:", prediction_label)

                    # This undoes vgg processing from stencil
                    mean = [103.939, 116.779, 123.68]
                    image[..., 0] += mean[0]
                    image[..., 1] += mean[1]
                    image[..., 2] += mean[2]
                    image = image[:, :, ::-1]
                    image = image / 255.
                    image = np.clip(image, 0., 1.)
                    
                    #shows the image so you can compare it to the predicted label in the terminal
                    # plt.imshow(image)
                    # plt.show()

                count += 1
            prediction = stats.mode(predictions)
            if (prediction[0].size!=0):
                predicted_label = datasets.idx_to_class[prediction[0][0]]
                print("Predicted label:", predicted_label)
                if (predicted_label == "del"):
                    labels = labels[:-1]
                elif (predicted_label == "space"):
                    labels = labels + " "
                elif (predicted_label != "nothing" and (len(labels)==0 or predicted_label != labels[-1])):
                    labels = labels + predicted_label
            return labels

        # Camera device
        # the argument is the device id. If you have more than one camera, you can access them by passing a different id, e.g., cv2.VideoCapture(1)
        self.vc = cv2.VideoCapture(0)
        if not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.use_camera = False

        if self.use_camera == False:
            # No camera!
            self.im = rgb2gray(img_as_float32(io.imread('images/YuanningHuCrop.png'))) 
        else:
            # We found a camera!
            # Requested camera size. This will be cropped square later on, e.g., 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Set the size of the output window
        cv2.namedWindow(self.wn, 0)
        fps = int(self.vc.get(cv2.CAP_PROP_FPS))
        save_interval = 1
        i = 0
        out_path = os.getcwd()+os.sep+'frame'+os.sep+'test'+os.sep+'A'
        labels = ""
        # Main loop
        while True:
            direct = os.listdir(out_path)
            a = time.perf_counter()
            self.run()
            ret, frame = self.vc.read()
            bbox, label, config = cv2.detect_common_objects(frame)
            output_image = cv2.draw_bbox(frame, bbox, label, config)
            cv2.imshow("Object Detection", output_image)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            i += 1
            if ret == False:
                break
            if i % (fps * save_interval) == 0:
                frame_name = 'Frame.jpg'
                cv2.imwrite(os.path.join(out_path, frame_name), frame)
            if len(direct) != 0:
                labels = classify_image(model, labels)
                print(labels)
                os.remove(out_path + '/' + 'Frame.jpg')
            # cv2.imshow('frame', frame); cv2.waitKey(0)
            # cv2.imwrite('test_frame.png', frame)
            # print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))
    
    
        if self.use_camera:
            # Stop camera
            self.vc.release()
    
    def run(self):
        
        if self.use_camera:
            # Read image. 
            # Some cameras will return 'None' on read until they are initialized, 
            # so sit and wait for a valid image.
            im = None
            while im is None:
                rval, im = self.vc.read()

        cv2.imshow("live cam", im) # faster alternative
        
        cv2.waitKey(1)

        return


if __name__ == '__main__':
    live()

