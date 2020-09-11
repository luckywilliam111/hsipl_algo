# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:32:48 2020

@author: user
"""

import numpy as np
from scipy.sparse import csr_matrix

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