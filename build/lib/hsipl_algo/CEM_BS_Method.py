# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:33:26 2020

@author: user
"""

import numpy as np
from scipy.stats import entropy

def BS_Corrcoef(imagecube, num):
    x, y, z = imagecube.shape
    
    hyperspectral = imagecube.reshape((x * y, z))
    scores = np.zeros((z, z))
    
    for i in range(z):
        for j in range(z):
            scores[i, j] = np.min(np.min(np.corrcoef(hyperspectral[:, i], hyperspectral[:, j])))
    
    hyperspectral_corrcoef = np.zeros((2, z))
    hyperspectral_corrcoef[0, :] = range(z)
    
    for i in range(z):
        hyperspectral_corrcoef[1, :] = np.sum(scores[i, :])-scores[i, i]
    
    for i in range(0, z):
        for j in range(0, z):
            if hyperspectral_corrcoef[1, i] > hyperspectral_corrcoef[1, j]:
                temp = hyperspectral_corrcoef[:, i].copy()
                hyperspectral_corrcoef[:, i] = hyperspectral_corrcoef[:, j].copy()
                hyperspectral_corrcoef[:, j] = temp.copy()
    
    band_select = hyperspectral_corrcoef[0, :num]
    
    return band_select

def BS_Entropy(imagecube, num):
    x, y, z = imagecube.shape
    
    hyperspectral = imagecube.reshape((x * y, z))
    hyperspectral_entropy = np.zeros((2, z))
    hyperspectral_entropy[0, :] = range(z)
    
    for i in range(z):
        hyperspectral_entropy[1, i] = entropy(hyperspectral[:, i])
    
    for i in range(0, z):
        for j in range(0, z):
            if hyperspectral_entropy[1, i] > hyperspectral_entropy[1, j]:
                temp = hyperspectral_entropy[:, i].copy()
                hyperspectral_entropy[:, i] = hyperspectral_entropy[:, j].copy()
                hyperspectral_entropy[:, j] = temp.copy()
    
    band_select = hyperspectral_entropy[0, :num]
    
    return band_select

def BS_STD(imagecube, num):
    x, y, z = imagecube.shape
    
    hyperspectral = imagecube.reshape((x * y, z))
    hyperspectral_std = np.zeros((2, z))
    hyperspectral_std[0, :] = range(0, z)
    
    for i in range(0, z):
        hyperspectral_std[1, i] = np.std(hyperspectral[:, i])
    
    for i in range(0, z):
        for j in range(0, z):
            if hyperspectral_std[1, i] > hyperspectral_std[1, j]:
                temp = hyperspectral_std[:, i].copy()
                hyperspectral_std[:, i] = hyperspectral_std[:, j].copy()
                hyperspectral_std[:, j] = temp.copy()
    
    band_select = hyperspectral_std[0, :num]
    
    return band_select

def CEM_BCC(imagecube, num): 
    xx, yy, band_num = imagecube.shape
    
    test_image = imagecube.reshape((xx * yy, band_num))
    test_image = np.mat(np.array(test_image))
    R = test_image * test_image.T / (xx * yy * 1.0)
    
    tt = np.mat(R) ** -1
    
    score = np.zeros(( band_num, band_num))
    
    for i in range(0,band_num):
        endmember_matrix = test_image[:, i]
        W = tt* endmember_matrix * ((endmember_matrix.T * tt * endmember_matrix) ** -1)
        
        for j in range(0, band_num):
            if i != j:
                test = test_image[:, j]
                score[i, j] = test.T * W
            else:
                score[i, j] = 1
    
    weight = np.zeros((band_num, 1))
    for i in range(0, band_num):
        test = score[i, :]
        scalar = np.sum(test) - score[i, i]
        weight[i] = scalar
        
    weight = np.abs(weight)
    original = range(0, band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num, 1))
    
    sorted_indices = np.argsort(-coefficient_integer, axis=0)
    original = np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    return band_select

def CEM_BCM(imagecube, num):
    xx, yy, band_num = imagecube.shape
    
    test_image = imagecube.reshape((xx * yy, band_num))
    test_image = np.mat(np.array(test_image))
    R = test_image * test_image.T / (xx * yy * 1.0)
    
    tt = np.mat(R) ** -1
    
    score = np.zeros((band_num, 1))
    
    for i in range(0, band_num):
        endmember_matrix = test_image[:, i]
        W = tt* endmember_matrix * ((endmember_matrix.T * tt * endmember_matrix) ** -1)
        score[i] = W.T * R * W
    
    weight = np.abs(score)
    original = range(0, band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num, 1))
    
    sorted_indices = np.argsort(-coefficient_integer, axis=0)
    original = np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    return band_select

def CEM_BDM(imagecube, num):
    xx, yy, band_num = imagecube.shape
    
    test_image = imagecube.reshape((xx * yy, band_num))
    test_image = np.mat(np.array(test_image))
    R = test_image * test_image.T / (xx * yy * 1.0)
    
    score = np.zeros((band_num, 1))
    
    for i in range(0, band_num):
        endmember_matrix = test_image[:, i]
        R_new = R - endmember_matrix * endmember_matrix.T
        R_new = R_new / (band_num -1)
        tt = np.mat(R_new) ** -1
        W = tt* endmember_matrix * ((endmember_matrix.T * tt * endmember_matrix) ** -1)
        score[i] = W.T * R * W
    
    weight = np.abs(score)
    original = range(0, band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num, 1))
    
    sorted_indices = np.argsort(-coefficient_integer, axis=0)
    original = np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    return band_select