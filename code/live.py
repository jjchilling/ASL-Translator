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
from scipy import ndimage
from skimage import io
from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray



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

        # Camera device
        # the argument is the device id. If you have more than one camera, you can access them by passing a different id, e.g., cv2.VideoCapture(1)
        self.vc = cv2.VideoCapture(0)
        if not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.use_camera = False

        if self.use_camera == False:
            # No camera!
            self.im = rgb2gray(img_as_float32(io.imread('images/YuanningHuCrop.png'))) # One of our intrepid TAs (Yuanning was one of our HTAs for Spring 2019)
        else:
            # We found a camera!
            # Requested camera size. This will be cropped square later on, e.g., 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Set the size of the output window
        cv2.namedWindow(self.wn, 0)

        # Main loop
        while True:
            a = time.perf_counter()
            self.run()
            print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))
    
    
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
