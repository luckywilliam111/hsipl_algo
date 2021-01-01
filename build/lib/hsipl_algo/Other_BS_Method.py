# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:14:38 2020

@author: user
"""

import numpy as np

def PCA_BS(im, num_band):
    im = (im.reshape(im.shape[2], im.shape[0] * im.shape[1])).transpose()
    
    cov_mat = np.cov(im.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    sort_band = np.argsort(eigen_vals)
    
    band_select = sort_band[:num_band]
    
    return band_select

def Uniform_BS(band_num, num):
    score = np.random.uniform(0, 1, [band_num, 1])
    weight = np.abs(score)
    original = range(0, band_num)
    coefficient_integer = weight * -1
    band_select = np.zeros((num, 1))
    
    sorted_indices = np.argsort(coefficient_integer, axis=0)
    original = np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    return band_select