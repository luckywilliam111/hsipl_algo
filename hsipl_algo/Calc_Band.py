# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:34:45 2020

@author: user
"""

import numpy as np
import scipy.special as ss

def HFC(HIM, t):
    x, y, z = HIM.shape
    pxl_no = x * y
    
    r = (HIM.reshape(x * y, z)).transpose()
    
    R = np.dot(r, np.transpose(r)) / pxl_no
    u = (np.mean(r, 1)).reshape(z, 1)
    K = R - np.dot(u, np.transpose(u))
    
    D1 = np.linalg.eig(R)
    D1 = np.sort(D1[0], 0)
    
    D2 = np.linalg.eig(K)
    D2 = np.sort(D2[0], 0)
    
    sita = np.sqrt(((D1 ** 2 + D2 ** 2) * 2) / pxl_no)
    
    P_fa = t
    
    Threshold = (np.sqrt(2)) * sita * ss.erfinv(1 - 2 * P_fa)
    
    Result = np.zeros([z, 1])
    
    for i in range(z):
        if (D1[i] - D2[i]) > Threshold[i]:
            Result[i] = 1
            
    number = int(np.sum(Result, 0))
    
    return number