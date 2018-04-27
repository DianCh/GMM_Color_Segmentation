#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:32:27 2018

@author: dianchen
"""

import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import analysis as als

# ===== Path Loading==========================
# No such folders in submitted code. Only used in training
folder_labels = "roipoly_annotate/labeled_data/RedBarrel/"
folder_pics = "roipoly_annotate/images/"
# ===== Path Loading==========================


def load_one_pic(picName, labelName):
# =============================================================================
#     # Input
#     # picName: file name of one image
#     # labelName: file name of one corresponding label
# =============================================================================
# =============================================================================
#     # Output
#     # X_mini: (3, N_mini) array of positive pixels
# =============================================================================
    
    # load data of one picture, using HSV space
    image = plt.imread(picName)
    image = clr.rgb_to_hsv(image)
    mask = np.load(labelName)
    
    # extract dimension information
    H, W = mask.shape
    
    if (H != image.shape[0] or W != image.shape[1]):
        print("Dimensions of image and mask don't match! Going to the next.")
        
        # return an empty mini set
        X_mini = np.zeros((3, 0))
        
        return X_mini
    
    # extract only the positive samples
    X_mini = image[mask].T
    
    return X_mini


def load_training_data(labelFolder=folder_labels, picFolder=folder_pics):
# =============================================================================
#     # Input
#     # labelFolder: folder under current directory for labels
#     # picFolder: folder under current directory for images
# =============================================================================
# =============================================================================
#     # Output
#     # X: (3, N) array of all positive pixels
# =============================================================================
    
    X = np.zeros((3, 0))
    Y = np.zeros((1, 0))
        
    # go over every training picture in specified folder
    for filename in os.listdir(picFolder):
        
        basename, extension = os.path.splitext(filename)
        # only considering ".png" format
        if (extension != ".png"):
            continue
        
        labelName = labelFolder + basename + ".npy"
        picName = picFolder + filename

        # check if corresponding label file exists
        if not os.path.isfile(labelName):
            print("This picture doesn't have labeled mask! Going to the next one.")
            continue
        
        # extract positive pixels from one picture
        X_mini = load_one_pic(picName, labelName)
        
        # concatenate to the batch
        X = np.concatenate((X, X_mini), axis=1)
        
    return X

def prepare_distance(picFolder=folder_pics):
# =============================================================================
#     # Input
#     # picFolder: folder containing training images
# =============================================================================
# =============================================================================
#     # Output
#     # X: array of labeled distances
#     # Y: array of calculated areas
# =============================================================================
    
    X = np.zeros((0, ))
    Y = np.zeros((0, ))
    
    # go over every training picture in specified folder
    for filename in os.listdir(picFolder):
        
        basename, extension = os.path.splitext(filename)
        # only considering ".png" format
        if (extension != ".png"):
            continue
        
        picName = picFolder + filename
        
        # get areas of barrels in one picture
        image = plt.imread(picName)
        _, _, props = als.detect_one_pic(image)
        
        X_temp = np.zeros((0, ))
        Y_temp = np.zeros((0, ))
        
        for one_prop in props:
            Y_temp = np.append(Y_temp, one_prop.area)
            
        # split file name to extract distance information
        string = basename.split(".")
        string = string[0]
        string = string.split("_")
        
        for s in string:
            X_temp = np.append(X_temp, float(s))
        
        # if number of areas and number of given distances don't match, skip this training example
        # because of incorrect prediction
        if X_temp.size != Y_temp.size:
            continue
        
        # sort distances and areas in opposite order to make them match
        X_temp[::-1].sort()
        Y_temp.sort()
        
        # append to the batch
        X = np.append(X, X_temp)
        Y = np.append(Y, Y_temp)
        
    return X, Y