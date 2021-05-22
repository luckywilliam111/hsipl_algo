# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 00:56:19 2021

@author: user
"""

import numpy as np
from sklearn.decomposition import FastICA

def ICA_BS(HIM, n_components, num):
    x, y, z = HIM.shape
    
    X = HIM.reshape(x * y, z)
    
    if int(n_components) > z:
        print('\033[31m'+"Error: The Number of components is higher than the number of bands in the data set")
        
    if int(num) > z:
        print('\033[31m'+"Warning: The Number of bands supposed to be selected is higher than the number of bands in the data set"+'\033[0m')
        
    ica = FastICA(n_components=int(n_components), whiten=True)
    
    S_ = ica.fit_transform(X)
    
    A_ = ica.mixing_
    
    W = np.linalg.pinv(A_)
        
    B_W = np.sum(np.absolute(W), axis=0)
    
    sortB_W = np.argsort(B_W)
    
    band_select = sortB_W[-int(num):]+1
    
    return band_select

def PCA_BS(im, num_band):
    im = (im.reshape(im.shape[2], im.shape[0] * im.shape[1])).transpose()
    
    cov_mat = np.cov(im.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    sort_band = np.argsort(eigen_vals)
    
    band_select = sort_band[:num_band]
    
    return band_select