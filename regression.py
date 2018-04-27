#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:54:45 2018

@author: dianchen
"""


import numpy as np
from scipy import stats

def train_regression(areas, distances):
# =============================================================================
#     # Input
#     # areas: array of areas from barrel detection
#     # distances: array of labeled distances
# =============================================================================
# =============================================================================
#     # Output
#     # slope: slope of linear regression
#     # intercept: intercept of linear regression
# =============================================================================
    
    one_over_area_sqrt = 1 / np.sqrt(areas)
    
    slope, intercept, _, _, _ = stats.linregress(one_over_area_sqrt, distances)
    
    return slope, intercept


def predict_one_distance(area, slope, intercept):
# =============================================================================
#     # Input
#     # area: calculated area from barrel detection
#     # slope: slope of linear regression
#     # intercept: intercept of linear regression
# =============================================================================
# =============================================================================
#     # Output
#     # distance: predicted distance
# =============================================================================
    
    one_over_area_sqrt = 1 / np.sqrt(area)
    
    distance = slope * one_over_area_sqrt + intercept
    
    return distance