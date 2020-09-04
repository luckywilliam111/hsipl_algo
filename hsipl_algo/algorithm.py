# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:38:34 2020

@author: user
"""

import numpy as np

def CEM(HIM, d):
    x, y, z = HIM.shape
    
    X = HIM.reshape(x * y, z)
    
    R = np.dot(np.transpose(X), X) / (x*y)
    IR = np.linalg.inv(R)
    
    A = (np.dot(X, np.dot(IR,d))) / (np.dot(np.transpose(d), np.dot(IR,d)))
    
    CEM_result = A.reshape(x, y)
    
    return CEM_result