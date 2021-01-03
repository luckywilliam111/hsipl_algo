# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:53:26 2020

@author: user
"""

import numpy as np
import pandas as pd

def BS_MaxV_BP(imagecube, d, num):
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        maxV_BP_band_select = MaxV_BP(imagecube, dd, num)
        maxV_BP_band_select = maxV_BP_band_select.reshape((maxV_BP_band_select.shape[0]), order='F')
        X.append(maxV_BP_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0] * X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    return band_select

def BS_MinV_BP(imagecube, d, num):
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        minV_BP_band_select = MinV_BP(imagecube, dd, num)
        minV_BP_band_select = minV_BP_band_select.reshape((minV_BP_band_select.shape[0]), order='F')
        X.append(minV_BP_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0] * X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    return band_select

def BS_SB_CTBS(imagecube, d, num):
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        SB_CTBS_band_select = SB_CTBS(imagecube, dd, num)
        SB_CTBS_band_select = SB_CTBS_band_select.reshape((SB_CTBS_band_select.shape[0]), order='F')
        X.append(SB_CTBS_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0] * X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    return band_select

def BS_SF_CTBS(imagecube, d, num):
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        SF_CTBS_band_select = SF_CTBS(imagecube, dd, num)
        SF_CTBS_band_select = SF_CTBS_band_select.reshape((SF_CTBS_band_select.shape[0]), order='F')
        X.append(SF_CTBS_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0] * X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    return band_select

def BmaxV_BP(imagecube, d, no_d, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    DU = np.hstack([d, no_d])
    c = np.vstack(([np.ones([d.shape[1], 1]), np.zeros([no_d.shape[1], 1])]))
    
    score=np.zeros((band_num,1))
    for i in range(0,band_num):
        T = np.delete(DU, i, 0)
        r = np.delete(test_image, i, 1)
        R = np.dot(np.transpose(r), r)/(xx*yy*1.0)
        tt = np.linalg.inv(R)
        W =  np.dot(np.dot(c.transpose(), 1 / (np.dot(np.dot(np.transpose(T), tt), T))), c)
        score[i] = W
        
    weight = np.abs(score)
    coefficient_integer = weight * -1
    
    sorted_y2 = np.argsort(coefficient_integer, axis=0)
    band_select = sorted_y2[:num]
    
    return band_select

def FminV_BP(imagecube, d, no_d, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    DU = np.hstack([d, no_d])
    c = np.vstack(([np.ones([d.shape[1], 1]), np.zeros([no_d.shape[1], 1])]))
    
    score=np.zeros((band_num, 1))
    for i in range(0,band_num):
        T = DU[i, :].reshape(1, DU.shape[1])
        r = test_image[:, i].reshape((test_image.shape[0], 1), order='F')
        R = np.dot(np.transpose(r), r)/(xx*yy*1.0)
        tt = np.linalg.inv(R)
        W =  np.dot(np.dot(c.transpose(), 1 / (np.dot(np.dot(T.transpose(), tt), T))), c)
        score[i] = W
    
    weight = np.abs(score)
    coefficient_integer = weight * 1
    
    sorted_y2 = np.argsort(coefficient_integer, axis=0)
    band_select = sorted_y2[:num]
    
    return band_select

def MaxV_BP(imagecube, d, num):
    xx, yy, band_num = imagecube.shape
    test_image = imagecube.reshape((xx * yy, band_num), order='F')
    
    score=np.zeros((band_num, 1))
    
    for i in range(0, band_num):
        d_new = np.delete(d, i, 0)
        r = np.delete(test_image, i, 1)
        R = np.dot(np.transpose(r), r) / (xx * yy * 1.0)
        tt = np.linalg.inv(R)
        W =  1 / ((np.dot(np.dot(np.transpose(d_new), tt), d_new)))
        score[i] = W
        
    weight = np.abs(score)
    original = range(0, band_num)
    coefficient_integer = weight * -1
    band_select = np.zeros((num, 1))
    
    sorted_indices = np.argsort(coefficient_integer, axis=0)
    original = np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    return band_select

def MinV_BP(imagecube, d, num):
    xx, yy, band_num = imagecube.shape
    test_image = imagecube.reshape((xx * yy, band_num), order='F')
    
    score=np.zeros((band_num, 1))
    for i in range(0, band_num):
        r = test_image[:, i].reshape((test_image.shape[0], 1), order='F')
        R = np.dot(np.transpose(r), r) / (xx * yy * 1.0)
        tt = np.linalg.inv(R)
        W =  1 / (np.dot(np.dot(np.transpose(d[i].reshape((1, 1), order='F')), tt), d[i].reshape((1, 1), order='F')))
        score[i] = W
    
    weight = np.abs(score)
    original = range(0, band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num, 1))
    
    sorted_indices = np.argsort(coefficient_integer, axis=0)
    original = np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    return band_select

def SB_CTBS(imagecube, d, num):
    xx, yy, band_num = imagecube.shape
    test_image = imagecube.reshape((xx * yy, band_num), order='F')
    
    max_band_select = MaxV_BP(imagecube, d, num)
    
    omega = []
    omega.append(np.int(max_band_select[0]))
    
    for i in range(0, num-1):
        score = np.zeros((band_num, 1))
        
        for j in range(0, band_num):
            bl = []
            bl.append(j)
            omega_bl = list(set(omega) | set(bl))
            
            new_d = np.delete(d, omega_bl, 0)
            new_r = np.delete(test_image, omega_bl, 1)
            
            new_R = np.dot(np.transpose(new_r), new_r) / (xx * yy * 1.0)
            new_tt = np.linalg.inv(new_R)
            new_W = 1 / (np.dot(np.dot(np.transpose(new_d), new_tt), new_d))
            score[j] = new_W
        
        weight = np.abs(score)
        coefficient_integer = weight * -1
        sorted_indices = np.argsort(coefficient_integer, axis=0)
        
        omega.append(np.int(sorted_indices[0]))
        
    band_select = np.array(omega)
    
    return band_select

def SB_TCIMBS(imagecube, d, no_d, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    DU = np.hstack([d, no_d])
    c = np.vstack(([np.ones([d.shape[1], 1]), np.zeros([no_d.shape[1], 1])]))
    
    Bmax_band_select = BmaxV_BP(imagecube, d, no_d, num)
    
    omega = []
    omega.append(np.int(Bmax_band_select[0]))
    
    for i in range(0, num-1):
        score = np.zeros((band_num,1))
        for j in range(0, band_num):
            bl = []
            bl.append(j)
            omega_bl = list(set(omega) | set(bl))
            
            T = np.delete(DU, omega_bl, 0)
            new_r = np.delete(test_image, omega_bl, 1)
            
            new_R = np.dot(np.transpose(new_r), new_r)/(xx*yy*1.0)
            new_tt = np.linalg.inv(new_R)
            new_W = np.dot(np.dot(c.transpose(), 1 / (np.dot(np.dot(T.transpose(), new_tt), T))), c)
            score[j] = new_W
        
        weight = np.abs(score)
        coefficient_integer = weight * -1
        sorted_indices = np.argsort(coefficient_integer, axis=0)
        
        omega.append(np.int(sorted_indices[0]))
        
    band_select = np.array(omega)
    
    return band_select

def SF_CTBS(imagecube, d, num):
    xx, yy, band_num = imagecube.shape
    test_image = imagecube.reshape((xx * yy, band_num), order='F')
    
    min_band_select = MinV_BP(imagecube, d, num)
    
    omega = []
    omega.append(np.int(min_band_select[0]))
    
    for i in range(0, num-1):
        score = np.zeros((band_num, 1))
        
        for j in range(0, band_num):
            new_d = []
            new_r = []
            bl = []
            bl.append(j)
            omega_bl = list(set(omega) | set(bl))
            
            for k in omega_bl:
                new_d.append(d[k])
                new_r.append(test_image[:, k])
                
            new_d = np.array(new_d)
            new_r = np.array(new_r)
            
            new_R = np.dot(new_r, np.transpose(new_r)) / (xx * yy * 1.0)
            new_tt = np.linalg.inv(new_R)
            new_W = 1 / (np.dot(np.dot(np.transpose(new_d), new_tt), new_d))
            score[j] = new_W
        
        weight = np.abs(score)
        coefficient_integer = weight * 1
        sorted_indices = np.argsort(coefficient_integer, axis=0)
        
        omega.append(np.int(sorted_indices[0]))
        
    band_select = np.array(omega)
    
    return band_select

def SF_TCIMBS(imagecube, d, no_d, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    DU = np.hstack([d, no_d])
    c = np.vstack(([np.ones([d.shape[1], 1]), np.zeros([no_d.shape[1], 1])]))
    
    Fmin_band_select = FminV_BP(imagecube, d, no_d, num)
    
    omega = []
    omega.append(np.int(Fmin_band_select[0]))
    
    for i in range(0, num-1):
        score = np.zeros((band_num,1))
        for j in range(0, band_num):
            bl = []
            bl.append(j)
            omega_bl = np.array(list(set(omega) | set(bl)))
            
            new_r = test_image[:, omega_bl].transpose()
            T = DU[omega_bl, :]
            
            new_R = np.dot(new_r, np.transpose(new_r))/(xx*yy*1.0)
            new_tt = np.linalg.inv(new_R)
            new_W = np.dot(np.dot(c.transpose(), 1 / (np.dot(np.dot(T.transpose(), new_tt), T))), c)
            score[j] = new_W
        
        weight = np.abs(score)
        coefficient_integer = weight * 1
        sorted_indices = np.argsort(coefficient_integer, axis=0)
        
        omega.append(np.int(sorted_indices[0]))
        
    band_select = np.array(omega)
    
    return band_select