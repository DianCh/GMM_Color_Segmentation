#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:13:30 2018

@author: dianchen
"""
import numpy as np
import scipy
from math import pi, exp
import matplotlib.pyplot as plt
import matplotlib.colors as clr


import data_loading as dl


def gauss_prob(mu, A, X):
# =============================================================================
#     # Input
#     # mu: (D, 1) array, D = 3 here
#     # A: (D, D) array, inverse of sigma
#     # X: (D, N) array, N = number of all data points to calculate
# =============================================================================
# =============================================================================
#     # Output
#     # p: (N, ) array indicating the probabilities of all data points
# =============================================================================
    
    # extract dimension
    D = X.shape[0]
    
    # calculate in advance the constant term
    a = (np.linalg.det(A) ** 0.5) / ((2 * pi) ** (D / 2))
    
    # compute powers of exponentials in fully vectorized form
    X_minus_mu = X - mu
    X_Cov = -0.5 * np.dot(X_minus_mu.T, A)
    
    powers = np.sum(np.multiply(X_Cov.T, X_minus_mu), axis=0)

    # compute exponentials
    b = np.exp(powers)
    
    # compute probabilities
    p = a * b
    
    return p

def gen_spd(D):
# =============================================================================
#     # Input
#     # D: an integer indicating the dimension to use
# =============================================================================
# =============================================================================
#     # Output
#     # A: (D, D) array
# =============================================================================
    
    # generate random sigma using U'XU
    d = np.random.rand(D)
    X = np.diag(d)
    U = scipy.linalg.orth(np.random.rand(D, D))
    sigma = np.dot(np.dot(U.T, X), U)
    
    return sigma

def est_param_gmm(X, num_Gaussian=3, thresh=0.005):
# =============================================================================
#     # Input
#     # X: (D, N) array, D = 3 here, N = number of all data points to calculate
#     # num_Gaussian: an integer, indicating the number of Gaussian mixture components to use
#     # thresh: hyperparameter that determines EM convergence
# =============================================================================
# =============================================================================
#     # Output
#     # mu: (D, num_Gaussian) array, estimated means
#     # sigma: (D, D, num_Gaussian) array, estimated covariances    
# =============================================================================
    
    # extract dimensions and number of examples
    dim, N = X.shape
    
    # compute ranges for each dimension
    range_0 = np.max(X[0, :]) - np.min(X[0, :])
    range_1 = np.max(X[1, :]) - np.min(X[1, :])
    range_2 = np.max(X[2, :]) - np.min(X[2, :])
    range_all = np.array([range_0, range_1, range_2])
    
    # compute data center
    center = np.sum(X, axis=1, keepdims=False) / N
    
    # initializations
    mu = np.zeros((dim, num_Gaussian))
    sigma = np.zeros((dim, dim, num_Gaussian))
    A = np.zeros((dim, dim, num_Gaussian))
    
    for k in range(num_Gaussian):
        mu[:, k] = np.random.rand(dim) * range_all + center
        sigma[:, :, k] = 10 * gen_spd(dim)
        
    g = np.zeros((N, num_Gaussian))
    z = np.random.rand(N, num_Gaussian)
    zk = np.sum(z, axis=0)
    
    # initialize variables for looping
    mu_prev = mu.copy()
    sigma_prev = sigma.copy()
    delta = np.zeros((num_Gaussian, 1))
    
    while True:

        # E Step: compute probabilities
        for k in range(num_Gaussian):
            # calculate in advance inverse of sigma
            A[:, :, k] = np.linalg.inv(sigma[:, :, k])
            # calculate probabilities in fully vectorized form
            g[:, k] = gauss_prob(mu[:, k, np.newaxis], A[:, :, k], X)
    
        # sum up probabilities from K mixture components
        g_sum = np.sum(g, axis=1, keepdims=True)
         
        # compute reletive weights z
        z = g / g_sum
        
        zk = np.sum(z, axis=0, keepdims=True)
        z = z / zk
        
        # M Step: update mu and sigma in fully vectorized form
        mu = np.dot(X, z)
        
        for k in range(num_Gaussian):
            X_minus_mu = X - mu[:, k, np.newaxis]
            X_minus_mu_weighted = X_minus_mu * (z[:, k, np.newaxis].T)
            
            sigma[:, :, k] = np.dot(X_minus_mu_weighted, X_minus_mu.T)
        
        # compute deltas to check for convergence
        delta_mu = mu - mu_prev
        delta_sigma = sigma - sigma_prev
        
        mu_prev = mu.copy()
        sigma_prev = sigma.copy()
       
        for k in range(num_Gaussian):
            delta[k, 0] = np.linalg.norm(delta_mu[:, k, np.newaxis]) + np.linalg.norm(delta_sigma[:, :, k], ord="fro")
            
        if (np.prod(delta < thresh) > 0):
            break
        
    print("mu: ", mu)
    
    return mu, sigma

        
def predict_pixels(mu, sigma, image):
# =============================================================================
#     # Input
#     # mu: (D, num_Gaussian) array, estimated means
#     # sigma: (D, D, num_Gaussian) array, estimated covariances    
#     # image: RGB image to predict
# =============================================================================
# =============================================================================
#     # Output
#     # mask: (H, W) logical array indicating the segmentation results, True for positive
# =============================================================================
    
    # extract dimensions and number of mixture components
    H, W, _ = image.shape
    K = mu.shape[1]
    
    # preprocess the RGB image, convert it to HSV and flip to needed shape
    image_hsv = clr.rgb_to_hsv(image)
    X = image_hsv.reshape((H * W, -1)).T

    # set thresh and initialize variables
    thresh = 6
    mask = np.zeros((H * W, 1))
    g = np.zeros((H * W, K))
    
    A = np.zeros(sigma.shape)
    for k in range(K):
        # calculate in advance inverse of sigma
        A[:, :, k] = np.linalg.inv(sigma[:, :, k])
        # calculate probabilities in fully vectorized form
        g[:, k] = gauss_prob(mu[:, k, np.newaxis], A[:, :, k], X)

    # sum up K mixture components
    mask = np.sum(g, axis=1, keepdims=True)
    # reshape the probs to rectangle shape        
    mask = mask.reshape((H, W))
    
    # thresh the probs to final predicted results
    return mask > thresh


def pick_params(num_Gaussian, X, thresh, num_sampling):
# =============================================================================
# # only helper function used by me during tuning    
# =============================================================================
    k1 = 0
    k2 = 0
    
    # inspect two frequent convergence and determine which one is optimal
    for i in range(num_sampling):
        mu, sigma = est_param_gmm(num_Gaussian, X, thresh)
        
        if np.sum(mu[0, :] > 0.9) == 2:
            k1 += 1
        
        if np.sum(np.multiply((mu[0, :] > 0.8), (mu[0, :] < 0.9))) == 1:
            k2 += 1
            
    print("k1: ", k1, "k2: ", k2)