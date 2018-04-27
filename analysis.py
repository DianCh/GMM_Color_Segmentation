#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:05:52 2018

@author: dianchen
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

import scipy.ndimage.morphology as mph
from skimage.measure import label, regionprops

import gmm
import regression as reg

# ===== Data & Path Loading==========================
mu = np.load("mu.npy")
sigma = np.load("sigma.npy")
slope = np.load("slope.npy")
intercept = np.load("intercept.npy")

#test_folder = "2018Proj1_test/"
test_folder = "roipoly_annotate/images/"

# ===== Data & Path Loading==========================

def postprocess_results(mask_raw):
# =============================================================================
#     # Input
#     # mask_raw: (H, W) logical array indicating predictions for pixels
# =============================================================================
# =============================================================================
#     # Output
#     # mask: (H, W) logical array indicating predictions for pixels
#     # props_selected: list of regions
# =============================================================================

    # morphological preprocessing
    mask = mph.binary_dilation(mask_raw, structure=np.ones((20,20)))
    mask = mph.binary_erosion(mask, structure=np.ones((40,40)))
    mask = mph.binary_dilation(mask, structure=np.ones((34, 34)))
    
    # get regions
    label_img = label(mask)
    props = regionprops(label_img)
    
    props_selected = []
    
    # filter regions by areas
    for one_prop in props:
        if one_prop.area > 2200:
            props_selected.append(one_prop)
    
    return mask, props_selected

    
def display_one_detection(image, props, save_path):
# =============================================================================
#     # Input
#     # image: RGB image to detect and display
#     # props: detected regions
# =============================================================================
# =============================================================================
#     # Output
#     # None
# =============================================================================
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for one_prop in props:
        
        # skip small regions
        if one_prop.area < 2200:
            continue
        
        # draw the centroid
        y0, x0 = one_prop.centroid
        ax.plot(x0, y0, '.g', markersize=15)
        
        # draw the bounding box
        minr, minc, maxr, maxc = one_prop.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)
        
        plt.title(save_path)
         
    plt.savefig(save_path)
    plt.show()
    

def detect_one_pic(image, save_path):
# =============================================================================
#     # Input
#     # image: RGB image
# =============================================================================
# =============================================================================
#     # Output
#     # mask_raw: (H, W) logical array indicating initial predictions
#     # mask: (H, W) logical array indicating predictions after postprocessing 
#     # props: list of regions
# =============================================================================
    
    # predict pixels of a picture using pre-trained parameters
    mask_raw = gmm.predict_pixels(mu, sigma, image)
    
    # get enhanced, final predictions
    mask, props = postprocess_results(mask_raw)
    
    # display the resulted figure
    display_one_detection(image, props, save_path)
    
    return mask_raw, mask, props


def predict_and_display_one(image_path, image_numer, save_path):
# =============================================================================
#     # Input
#     # image_path: complete image file name
#     # image_number: current number of this image
# =============================================================================
# =============================================================================
#     # Output
#     # mask_raw: (H, W) logical array indicating initial predictions
#     # mask: (H, W) logical array indicating predictions after postprocessing 
#     # props: list of regions
#     # distances: list of distance predictions
# =============================================================================
    
    # read in image
    image = plt.imread(image_path)
    
    # get detection results
    mask_raw, mask, props = detect_one_pic(image, save_path)
    distances = []
    
    print("For Image", image_numer)
    
    # check if no barrels are detected
    if len(props) == 0:
        
        print("No barrels detected in this image!")
        return mask_raw, mask, props, distances
    
    # predict distances and display results
    count = 1
    for one_prop in props:
        
        y0, x0 = one_prop.centroid
        area = one_prop.area
        
        distance = reg.predict_one_distance(area, slope, intercept)
        distances.append(distance)
        
        print("Barrel", count, "(centroidX, centroidY) =", (x0, y0), "distance =", distance)
        
        count += 1
        
    return mask_raw, mask, props, distances


def predict_and_display_all(picFolder=test_folder):
# =============================================================================
#     # Input
#     # picFolder: folder under Project_1 that stores test images
# =============================================================================
# =============================================================================
#     # Output
#     # results_all: compact results for all tested images
# =============================================================================
    
    # initialize result list
    results_all = []
    
    # initialize counter
    count = 1
    
    for filename in os.listdir(picFolder):
        
        # get predictions of one image
        image_path = picFolder + filename
        save_path =  filename
        mask_raw, mask, props, distances = predict_and_display_one(image_path, count, save_path)
        
        # count up by 1 and append this result to the list
        count += 1
        results_all.append([mask_raw, mask, props, distances])
        
        # wait for keypress to go on to the next test image
        input("Press ENTER to continue")

    return results_all


if __name__ == '__main__':
    
    input("Press ENTER to start testing images")
    
    results_all = predict_and_display_all(test_folder)
    