# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:07:14 2020

@author: user
"""

import numpy as np
import Target_Algorithm as TA

def ED_d(HIM, target):
    x, y, z = HIM.shape
    
    B = np.transpose(HIM.reshape(x * y, z))
    
    ED_result = np.sqrt(np.sum((B - target) * (B - target), 0))
    
    return ED_result

def SID_d(cutba, target):
    x, y, z = cutba.shape
    
    B = np.transpose(cutba.reshape(x * y, z))
    
    B[B == 0] = 0.01
    
    pl = B / (np.sum(B, axis=0))
    ql = target / (np.sum(target, axis=0))
    
    Dsisj = pl * (np.log(pl / ql))
    Dsisj = np.sum(Dsisj, axis=0)
    Dsjsi = ql * (np.log(ql / pl))
    Dsjsi = np.sum(Dsjsi, axis=0)
    SID_result = Dsisj + Dsjsi
    
    return SID_result

def weight_ED_CEM(HIM, d):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    ED_weight = (ED_d(HIM, d)).reshape(x * y, 1)
    
    R = np.dot(X, ED_weight * X.transpose()) / (x * y)
    
    IR = np.linalg.inv(R)
    
    A = (np.dot(np.transpose(X), np.dot(IR, d))) / (np.dot(np.transpose(d), np.dot(IR, d)))
    
    weight_ED_CEM_result = A.reshape(x, y)
    
    return weight_ED_CEM_result

def weight_SID_CEM(HIM,d):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    SID_weight = (1 - SID_d(HIM, d)).reshape(x * y, 1)
    
    R = np.dot(X, SID_weight * X.transpose()) / (x * y)
    
    IR = np.linalg.inv(R)
    
    A = (np.dot(np.transpose(X), np.dot(IR, d))) / (np.dot(np.transpose(d), np.dot(IR, d)))
    
    weight_SID_CEM_result = A.reshape(x, y)
    
    return weight_SID_CEM_result

def Winner_Take_All_CEM(HIM, d):
    CEM_result = TA.CEM(HIM, d)
    
    Winner_Take_All_CEM_result = 1 - np.power(CEM_result, 2)
    
    return Winner_Take_All_CEM_result

def weight_Winner_Take_All_CEM(HIM, d):
    x, y, z = HIM.shape
    
    X = np.transpose(HIM.reshape(x * y, z))
    
    Winner_Take_All_CEM_result = (Winner_Take_All_CEM(HIM, d)).reshape(x * y, 1)
    
    R = np.dot(X, Winner_Take_All_CEM_result * X.transpose()) / (x * y)
    
    IR = np.linalg.inv(R)
    
    A = (np.dot(np.transpose(X), np.dot(IR, d))) / (np.dot(np.transpose(d), np.dot(IR, d)))
    
    weight_Winner_Take_All_CEM_result = A.reshape(x, y)
    
    return weight_Winner_Take_All_CEM_result