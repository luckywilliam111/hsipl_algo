# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:32:48 2020

@author: user
"""

import numpy as np
import numpy.matlib as mb
from scipy.sparse import csr_matrix

def GA(M):
    L = algo(M)
    
    new_min = np.min(M[:])
    new_max = np.max(M[:])
    
    L = nma_rescale(L, new_min, new_max)
    
    L = mb.repmat(L, 1, M.shape[1])
    
    S = M - L
    
    return L, S

def Godec(data):
    x, y = data.shape
    
    rank = 1
    card = x * y
    power = 0
    iter_max = 1e+2
    error_bound = 1e-3
    iterate = 1
    
    RMSE = []
    
    if x < y:
        data = data.transpose()
        
    L = data.copy()
    S = csr_matrix(np.zeros([x, y])).toarray()
    
    while True:
        Y2 = np.random.randn(y, rank)
        
        for i in range(power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.transpose(), Y1)
            
        Q, R = np.linalg.qr(Y2)
        
        L_new = np.dot(np.dot(L, Q), Q.transpose())
        
        T = L - L_new + S
        
        L = L_new.copy()
        
        idx = (np.argsort(-1 * abs(T.reshape(1, x * y)))).reshape(x * y)
        
        S = np.zeros([x * y])
        
        S[idx[0:card]] = T.reshape(x * y)[idx[0:card]]
        
        S = S.reshape(x, y)
        
        T.reshape(x * y)[idx[0:card]] = 0
        
        RMSE.append(np.linalg.norm(T.reshape(x * y)))
        
        if (RMSE[-1] < error_bound) or (iterate > iter_max):
            break
        else:
            L = L + T
            
        iterate = iterate + 1
        
    LS = L+S
    
    error = np.linalg.norm(LS.reshape(x * y) - data.reshape(x * y)) / np.linalg.norm(data.reshape(x * y))
    
    if x < y:
        LS = LS.transpose()
        L = L.transpose()
        S = S.transpose()
        
    return L, S

def algo(data):
    X = data.transpose()

    K = 1
    epsilon = 10 * np.finfo(float).eps
    
    N, D = X.shape
    
    vectors = np.zeros([D, K])
    
    vectors[:] = np.NAN
    
    for k in range(K):
        mu = np.random.rand(D, 1) - 0.5
        
        mu = mu / np.linalg.norm(mu)
        
        for iterate in range(3):
            dots = np.dot(X, mu)
            mu = (np.dot(dots.transpose(), X)).transpose()
            mu = mu / np.linalg.norm(mu)
            
        for iterate in range(N):
            prev_mu = mu.copy()
            dot_signs = np.sign(np.dot(X, mu))
            mu = np.dot(dot_signs.transpose(), X)
            mu = (mu / np.linalg.norm(mu)).transpose()
            
            if np.max(abs(mu - prev_mu)) < epsilon:
                break
            
        if k == 0:
            vectors[:, k] = mu.reshape(D)
            X = X - np.dot(np.dot(X, mu), mu.transpose())
    
    return vectors

def nma_rescale(A, new_min, new_max):
    current_max = np.max(A[:])
    current_min = np.min(A[:])
    C =((A - current_min) * (new_max - new_min)) / (current_max - current_min) + new_min
    
    return C