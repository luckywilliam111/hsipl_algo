# -*- coding: utf-8 -*-
"""
Created on Sat May  8 01:25:33 2021

@author: WEN
"""

import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        
    def fit(self, data):
        data_T = data.transpose()
        
        L, K = data_T.shape
        
        self.m = (np.mean(data_T, 1)).reshape(L, 1)
        
        C = np.cov(data_T - self.m)
        
        self.eigen_vals, self.eigen_vecs = np.linalg.eigh(C)
        
        self.Index = np.argsort(self.eigen_vals)[::-1]
        
        V_sort = self.eigen_vecs[:, self.Index].transpose()
        
        self.V = V_sort[:self.n_components]
        
        self.components_ = self.V
        
        self.explained_variance = self.eigen_vals[self.Index]
        
        self.explained_variance_ratio_ = self.explained_variance / np.sum(self.eigen_vals)
     
    def transform(self, data):
        data_hat = data - self.m.T
        
        PCs = np.dot(data_hat, self.V.T)
        
        return PCs
    
    def get_SelectBand(self):
        return self.Index[:self.n_components]