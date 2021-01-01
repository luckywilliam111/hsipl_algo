# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 03:01:25 2020

@author: user
"""

import numpy as np

def K_RXD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x * y)
	
    iK = np.linalg.inv(K)
    
    dr = np.sum((np.dot(Bu, iK)) * Bu, 1)
    
    K_RXD_result = dr.reshape(x, y)
    
    return K_RXD_result

def LPTD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    
    R = np.dot(np.transpose(B), B) / (x * y)
    
    iR = np.linalg.inv(R)
    
    temp = np.ones([1, z])
	
    dr = np.sum(np.dot(temp, iR) * B, 1)
    
    dr = 1 - dr / np.max(dr)
    
    LPTD_result = dr.reshape(x, y)
    
    return LPTD_result

def M_RXD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x * y)
    
    iK = np.linalg.inv(K)
    
    C = np.power(np.transpose(np.sum(Bu * Bu, 1)), 0.5)
    
    dr = C * np.sum((np.dot(Bu, iK)) * Bu, 1)
    
    M_RXD_result = dr.reshape(x, y)
    
    return M_RXD_result

def N_RXD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x * y)
    
    iK = np.linalg.inv(K)
    
    C = np.power(np.transpose(np.sum(Bu * Bu, 1)), -1)
    
    dr = C * np.sum((np.dot(Bu, iK)) * Bu, 1)    
    
    N_RXD_result = dr.reshape(x, y)
    
    return N_RXD_result

def R_RXD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    
    R = np.dot(np.transpose(B), B) / (x * y)
    
    iR = np.linalg.inv(R)
    
    dr = np.sum(((np.dot(B, iR)) * B), 1)
    
    R_RXD_result = dr.reshape(x, y)
    
    return R_RXD_result

def UTD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x * y)
    
    iK = np.linalg.inv(K)
    
    temp = np.ones([1, z]) - rep_u
    
    dr = np.sum(np.dot(temp, iK) * Bu, 1)
    
    dr = 1 - dr / np.max(dr)
    
    UTD_result = dr.reshape(x, y)
    
    return UTD_result

def UTD_RXD(original):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x * y)
    
    iK = np.linalg.inv(K)
        
    B1 = B - 1
    
    dr = np.sum(np.dot(B1, iK) * Bu, 1)   
    
    UTD_RXD_result = dr.reshape(x, y)
    
    return UTD_RXD_result