# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:40:14 2020

@author: user
"""

import numpy as np

def awgn(x, SNR):
    SNR = 10 ** (SNR / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / SNR
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    
    return x + noise

def hKACE(HIM, d, SNR, lamb, epsilon, max_iter, add_noise):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    if add_noise == True:
        for i in range(x * y):
            X[:, i] = awgn(X[:, i], SNR)
    
    Weight = np.ones([1, x * y])
    hACEMap_old = np.ones([1, x * y])
    
    Energy = []
    
    for i in range(max_iter):
        X = X * Weight
        
        u = np.mean(np.transpose(X), 0)
        
        rep_u = u.reshape(z, 1)
        
        K = np.dot(X - rep_u, np.transpose(X - rep_u)) / (x * y)
        
        iK = np.linalg.inv(K + 0.0001 * np.eye(z))
        
        hACEMap = np.power(np.dot(np.dot(d.transpose(), iK), X), 2) / (np.dot(np.dot(d.transpose(), iK), d) * (np.sum(np.dot(X.transpose(), iK) * X.transpose(), 1)).reshape(1, x*y))
        
        Weight = 1 - np.power(2.71828, (-lamb * hACEMap))
        
        Weight[Weight < 0] = 0
        
        res = np.power(np.linalg.norm(hACEMap_old), 2) / (x * y) - np.power(np.linalg.norm(hACEMap), 2) / (x * y)
        
        Energy.append(np.power(np.linalg.norm(hACEMap), 2) / (x * y))
        
        hACEMap_old = hACEMap.copy()
        
        if abs(res) < epsilon:
            break
    
    hACEMap = hACEMap.reshape(x, y)
    
    return hACEMap

def hAMF(HIM, d, SNR, lamb, epsilon, max_iter, add_noise):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    if add_noise == True:
        for i in range(x * y):
            X[:, i] = awgn(X[:, i], SNR)
    
    Weight = np.ones([1, x * y])
    hAMFMap_old = np.ones([1, x * y])
    
    Energy = []
    
    for i in range(max_iter):
        X = X * Weight
        
        u = np.mean(np.transpose(X), 0)
        
        rep_u = u.reshape(z, 1)
        
        K = np.dot(X - rep_u, np.transpose(X - rep_u)) / (x * y)
        
        iK = np.linalg.inv(K + 0.0001 * np.eye(z))
        
        hAMFMap = np.dot(np.dot(d.transpose(), iK), X) / np.dot(np.dot(d.transpose(), iK), d)
        
        Weight = 1 - np.power(2.71828, (-lamb * hAMFMap))
        
        Weight[Weight < 0] = 0
        
        res = np.power(np.linalg.norm(hAMFMap_old), 2) / (x * y) - np.power(np.linalg.norm(hAMFMap), 2) / (x * y)
        
        Energy.append(np.power(np.linalg.norm(hAMFMap), 2) / (x * y))
        
        hAMFMap_old = hAMFMap.copy()
        
        if abs(res) < epsilon:
            break
    
    hAMFMap = hAMFMap.reshape(x, y)
    
    return hAMFMap

def hCEM(HIM, d, SNR, lamb, epsilon, max_iter, add_noise):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    if add_noise == True:
        for i in range(x * y):
            X[:, i] = awgn(X[:, i], SNR)
    
    Weight = np.ones([1, x * y])
    hCEMMap_old = np.ones([1, x * y])
    
    Energy = []
    
    for i in range(max_iter):
        X = X * Weight
            
        R = np.dot(X, X.transpose()) / (x * y)
        
        iR = np.linalg.inv(R + 0.0001 * np.eye(z))
        
        w = np.dot(iR, d) / np.dot(np.dot(d.transpose(), iR), d)
        
        hCEMMap = np.dot(w.transpose(), X)
        
        Weight = 1 - np.power(2.71828, (-lamb * hCEMMap))
        
        Weight[Weight < 0] = 0
        
        res = np.power(np.linalg.norm(hCEMMap_old), 2) / (x * y) - np.power(np.linalg.norm(hCEMMap), 2) / (x * y)
        
        Energy.append(np.power(np.linalg.norm(hCEMMap), 2) / (x * y))
        
        hCEMMap_old = hCEMMap.copy()
        
        if abs(res) < epsilon:
            break
        
    hCEMMap = hCEMMap.reshape(x, y)
    
    return hCEMMap

def hKMD(HIM, d, SNR, lamb, epsilon, max_iter, add_noise):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    if add_noise == True:
        for i in range(x * y):
            X[:, i] = awgn(X[:, i], SNR)
    
    Weight = np.ones([1, x * y])
    hKMDMap_old = np.ones([1, x * y])
    
    Energy = []
    
    for i in range(max_iter):
        X = X * Weight
        
        u = np.mean(np.transpose(X), 0)
        
        rep_u = u.reshape(z, 1)
        
        K = np.dot(X - rep_u, np.transpose(X - rep_u)) / (x * y)
        
        iK = np.linalg.inv(K + 0.0001 * np.eye(z))
        
        hKMDMap = (np.sqrt(np.sum((np.dot(np.transpose(X - d), iK)) * (X - d).transpose(), 1))).reshape(1, x * y)
        
        hKMDMap = 1 - (hKMDMap / np.max(hKMDMap))
        
        Weight = 1 - np.power(2.71828, (-lamb * hKMDMap))
        
        Weight[Weight < 0] = 0
        
        res = np.power(np.linalg.norm(hKMDMap_old), 2) / (x * y) - np.power(np.linalg.norm(hKMDMap), 2) / (x * y)
        
        Energy.append(np.power(np.linalg.norm(hKMDMap), 2) / (x * y))
        
        hKMDMap_old = hKMDMap.copy()
        
        if abs(res) < epsilon:
            break
    
    hKMDMap = hKMDMap.reshape(x, y)
    
    return hKMDMap