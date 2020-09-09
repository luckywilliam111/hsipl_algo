# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:38:34 2020

@author: user
"""

import numpy as np

def ACE(original, target):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x*y)
    
    iK = np.linalg.inv(K)
    
    dr = np.power(np.dot(np.dot(target.transpose(), iK), B.transpose()), 2) / (np.dot(np.dot(target.transpose(), iK), target) * (np.sum(np.dot(B, iK) * B, 1)).reshape(1, x*y))
	
    ACE_result = dr.reshape(x, y)
    
    return ACE_result

def AMF(original, target):
    x, y, z = original.shape

    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x*y)
    
    iK = np.linalg.inv(K)
    
    dr = np.dot(np.dot(target.transpose(), iK), B.transpose()) / np.dot(np.dot(target.transpose(), iK), target)
    
    AMF_result = dr.reshape(x, y)
    
    return AMF_result

def ASW_CEM(HIM, d, Sprout_HIM, minwd, midwd, maxwd, wd_range, sprout_rate):
    x, y, z = HIM.shape
    
    wd_matrix = np.zeros([x, y])
    mid_ASW_CEM_result = np.zeros([x, y])
    K = midwd
    
    for i in range(x):
        j = 0
        countnum = 0
        while j < y:
            half = np.fix(K / 2)
            
            x1 = np.int(i - half)
            x2 = np.int(i + half)
            y1 = np.int(j - half)
            y2 = np.int(j + half)
            
            if x1 <= 0:
                x1 = 0
            elif x2 >= x:
                x2 = x
                
            if y1 <= 0:
                y1 = 0
            elif y2 >= y:
                y2 = y
            
            xx, yy, zz = Sprout_HIM.shape
            Sprout = Sprout_HIM.reshape(xx * yy, zz)
            temp = np.sum(np.sum(Sprout[x1:x2, y1:y2], 0), 0)
            sumsprout = temp
            num = Sprout[x1:x2, y1:y2].shape[0] * Sprout[x1:x2, y1:y2].shape[1]
            
            if (sumsprout / num) < (sprout_rate - 0.001) and countnum == 0:
                K = K - wd_range
                j = j - 1
                countnum = 1
            elif (sumsprout / num) > (sprout_rate + 0.001) and countnum == 0:
                K = K + wd_range
                j = j - 1
                countnum = 2;
            elif (sumsprout / num) < (0.01) and countnum == 1 and K > minwd:
                K = K - wd_range
                j = j - 1
            elif (sumsprout / num) > (0.01) and countnum == 2 and K < maxwd:
                K = K + wd_range
                j = j - 1
            else:
                Local_HIM = HIM[x1:x2, y1:y2,:]
                
                xxx, yyy, zzz = Local_HIM.shape
                X = Local_HIM.reshape(xxx * yyy, zzz)
                S = np.dot(np.transpose(X), X)
                r = np.reshape(HIM[i, j, :], [z, 1])
				
                IS = np.linalg.inv(S)
                
                mid_ASW_CEM_result[i, j] = np.dot(np.dot(np.transpose(r), IS), d) / np.dot(np.dot(np.transpose(d), IS), d)
                wd_matrix[i, j] = K
                K = midwd
                countnum = 0
            j = j + 1
    
    return mid_ASW_CEM_result

def CBD(HIM, target):
    x, y, z = HIM.shape
    
    X = HIM.reshape(x * y, z)
    
    dr = np.sum(abs(X - target.transpose()), 1)
    
    dr = 1 - (dr / np.max(dr))
        
    CBD_result = dr.reshape(x, y)
    
    return CBD_result

def CEM(HIM, d):
    x, y, z = HIM.shape
    
    X = HIM.reshape(x * y, z)
    
    R = np.dot(np.transpose(X), X) / (x * y)
    IR = np.linalg.inv(R)
    
    A = (np.dot(X, np.dot(IR, d))) / (np.dot(np.transpose(d), np.dot(IR, d)))
    
    CEM_result = A.reshape(x, y)
    
    return CEM_result

def ED(original, target):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    
    Bt = B - target.transpose()
    
    dr = np.sqrt(np.sum(Bt * Bt, 1))
    
    dr = 1 - (dr / np.max(dr))
    
    ED_result = dr.reshape(x, y)
    
    return ED_result

def GLRT(original, target, Non_target):
    x, y, z = original.shape
    
    B = (original.reshape(x * y, z)).transpose()
    
    PB = np.eye(Non_target.shape[0]) - np.dot(np.dot(Non_target, np.dot(Non_target.transpose(), Non_target)), Non_target.transpose())
    
    dU = np.hstack([target, Non_target])
    
    PSB = np.eye(Non_target.shape[0]) - np.dot(np.dot(dU, np.dot(dU.transpose(), dU)), dU.transpose())
    
    gl = np.sum(np.dot(B.transpose(), PB - PSB) * B.transpose(), 1) / np.sum(np.dot(B.transpose(), PSB) * B.transpose(), 1)
    
    GLRT_result = gl.reshape(x, y)
    
    return GLRT_result

def JMD(HIM, target):
    x, y, z = HIM.shape
    
    X = (HIM.reshape(x * y, z)).transpose()
    
    pl = X / (np.sum(X, 0))
    ql = target / (np.sum(target, 0))
        
    dr = np.sqrt(np.sum(np.power((np.sqrt(pl) - np.sqrt(ql)), 2), 0))
    
    dr = 1 - (dr / np.max(dr))
        
    JMD_result = dr.reshape(x, y)
     
    return JMD_result

def KMD(original, target):
    x, y, z = original.shape

    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x*y)
    
    iK = np.linalg.inv(K)
    
    BD = B - target.transpose()
    
    dr = np.sqrt(np.sum((np.dot(BD, iK)) * BD, 1))
        
    dr = 1 - (dr / np.max(dr))
    
    KMD_result = dr.reshape(x, y)
    
    return KMD_result

def KMFD(original, target):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x*y)
    
    iK = np.linalg.inv(K)
    
    du = target.transpose() - rep_u
    
    dr = np.dot(np.dot(Bu, iK), du.transpose())
        
    KMFD_result = dr.reshape(x, y)
    
    return KMFD_result

def OPD(HIM, d):
    x, y, z = HIM.shape
    
    X = (HIM.reshape(x * y, z)).transpose()
    
    I = np.eye(z, z)
    P = I - np.dot(np.dot(d, np.linalg.inv(np.dot(np.transpose(d), d))), np.transpose(d))
    
    dr = np.sqrt(np.sum(np.dot(np.transpose(X), P) * X.transpose(), 1) + np.dot(np.dot(np.transpose(d), P), d))
    
    dr = 1 - (dr / np.max(dr))
    
    OPD_result = dr.reshape(x, y)
    
    return OPD_result

def OSP(original, target, Non_target):
    x, y, z = original.shape
    
    B = (original.reshape(x * y, z)).transpose()
    I = np.eye(z)
    
    P = I - np.dot(np.dot(Non_target, (np.power(np.dot(np.transpose(Non_target), Non_target), -1))), np.transpose(Non_target))
    
    dr = np.dot(np.dot(np.transpose(target), P), B)
    
    OSP_result = dr.reshape(x, y)
    
    return OSP_result

def RMD(original, d):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    
    R = np.dot(np.transpose(B), B) / (x*y)
    
    iR = np.linalg.inv(R)
    
    Bd = B - d.transpose()
    
    dr = np.sqrt(np.sum(((np.dot(Bd, iR)) * Bd), 1))
    
    dr = 1 - (dr / np.max(dr))
    
    RMD_result = dr.reshape(x, y)
    
    return RMD_result

def RMFD(original, target):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    
    R = np.dot(np.transpose(B), B) / (x*y)
    
    IR = np.linalg.inv(R)
        
    dr = np.dot(np.dot(np.transpose(target), IR), B.transpose())
        
    RMFD_result = dr.reshape(x, y)
    
    return RMFD_result

def SAM(original, target):
    x, y, z = original.shape
    
    B = (original.reshape(x * y, z)).transpose()
    
    inner_ori_target = (B * target).sum(axis=0)
    norm_ori = np.power(np.power(B, 2).sum(axis=0), 0.5)
    norm_target = np.power(np.power(target, 2).sum(axis=0), 0.5)
    x2 = inner_ori_target / (norm_ori * norm_target)
    
    dr = np.arccos(abs(x2))
    
    SAM_result = 1 - dr.reshape(x, y)
    
    return SAM_result

def SID(cutba, target):
    x, y, z = cutba.shape
    
    cutba[cutba == 0] = 0.1
    
    B = (cutba.reshape(x * y, z)).transpose()
    
    pl = B / (np.sum(B, axis=0))
    ql = target / (np.sum(target, axis=0))
    
    Dsisj = pl * (np.log(pl / ql))
    Dsisj = np.sum(Dsisj, axis=0)
    Dsjsi = ql * (np.log(ql / pl))
    Dsjsi = np.sum(Dsjsi, axis=0)
    
    dr = Dsisj + Dsjsi
    
    dr = 1 - (dr / np.max(dr))
    
    SID_result = dr.reshape(x, y)
    
    return SID_result

def SID_sin_SAM(HIM, target):
    SID_sin_SAM_result = SID(HIM, target) * np.sin(SAM(HIM, target))
    
    return SID_sin_SAM_result

def SID_tan_SAM(HIM, target):
    SID_tan_SAM_result = SID(HIM, target) * np.tan(SAM(HIM, target))
    
    return SID_tan_SAM_result

def SMF(original, target):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x*y)
    
    iK = np.linalg.inv(K)
    
    tu = target - rep_u.transpose()
    temp1 = np.dot(np.dot(np.transpose(tu), iK), tu)
    
    dr = np.dot(np.dot(Bu, iK), tu) / temp1
    
    SMF_result = dr.reshape(x, y)
    
    return SMF_result

def Subset_CEM(HIM, d, w, h):
    x, y, z = HIM.shape
    
    Subset_CEM_result = np.zeros([x, y])

    for i in range(0, x, w):
        for j in range(0, y, h):
            Subset_CEM_result[i:i+(w), j:j+(h)] = CEM(HIM[i:i+(w), j:j+(h), :], d)
    
    return Subset_CEM_result

def SW_CEM(HIM, d, K):
    x, y, z = HIM.shape
    
    half = np.fix(K / 2);
    SW_CEM_result = np.zeros([x, y])
    
    for i in range(x):
        for j in range(y):
            x1 = np.int(i - half)
            x2 = np.int(i + half)
            y1 = np.int(j - half)
            y2 = np.int(j + half)
            
            if x1 <= 0:
                x1 = 0;
            elif x2 >= x:
                x2 = x
                
            if y1 <= 0:
                y1 = 0;
            elif y2 >= y:
                y2 = y
            
            Local_HIM = HIM[x1:x2, y1:y2, :]
            
            xx, yy, zz = Local_HIM.shape
            X = Local_HIM.reshape(xx * yy, zz)
            S = np.dot(np.transpose(X), X)
            r = np.reshape(HIM[i, j, :], [z, 1])
            
            IS = np.linalg.inv(S)
         
            SW_CEM_result[i, j] = np.dot(np.dot(np.transpose(r), IS), d) / np.dot(np.dot(np.transpose(d), IS), d)
    
    return SW_CEM_result

def TCIMF(original, target, Non_target):
    x, y, z = original.shape
    
    B = original.reshape(x * y, z)
	
    R = np.dot(np.transpose(B), B) / (x*y)
	
    IR = np.linalg.inv(R)
    
    DU = np.hstack([target, Non_target])
    zz = np.hstack(([np.ones([1, target.shape[1]]), np.zeros([1, Non_target.shape[1]])]))

    temp = np.dot(np.dot(np.dot(np.linalg.inv(R), DU), np.dot(np.dot(np.transpose(DU), IR), DU)), np.transpose(zz))

    dr = np.dot(B, temp)
    
    TCIMF_result = dr.reshape(x, y)
    
    return TCIMF_result

def TD(HIM, target):
    x, y, z = HIM.shape
    
    X = (HIM.reshape(x * y, z)).transpose()
    
    dr = np.amax(abs(X - target), 0)
    
    dr = 1 - (dr / np.max(dr))
        
    TD_result = dr.reshape(x, y)
    
    return TD_result