# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:32:48 2020

@author: user
"""

import numpy as np
import numpy.matlib as mb
from scipy.sparse import csr_matrix

def GA(M):
    L = GA_algo(M)
    
    new_min = np.min(M[:])
    new_max = np.max(M[:])
    
    L = nma_rescale(L, new_min, new_max)
    
    L = mb.repmat(L, 1, M.shape[1])
    
    S = M - L
    
    return L, S

def GA_algo(data):
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

def OPRMF(data):
    X = normalize(data)
    
    L = OPRMF_algo(X)
    
    S = data - L
    
    return L, S

def OPRMF_algo(X):
    rk = 2
    lambdaU = 1
    lambdaV = 1
    tol = 1e-2
    
    x, y = X.shape
    
    mask = np.ones([x, y])
    
    maxIter = 40
    
    startIndex = 1
    
    U = np.random.randn(x, rk)
    
    V = np.random.randn(rk, startIndex)
    
    lambd = 1
    eps = 1e-3
    
    IS = csr_matrix(np.eye(rk)).toarray()
    
    A = []
    B = []
    
    TA = []
    TB = []
    
    forgetFactor = 0.98
    confidence = 1e-3
    
    L = np.zeros([x, y])
    
    for j in range(y):
        Y = X[:, 0:j+1]
        
        if j != 0:
           V = np.hstack([V, V[:, j-1].reshape(V.shape[0], 1)])
           
        r = abs(Y - np.dot(U, V))
        
        confidence = np.min(confidence * 1.5)
        
        c = 0
        
        while True:
            c = c + 1
            
            oldR = r.copy()
            
            r = abs(Y - np.dot(U, V))
            
            r = (r < eps).astype(np.int) * eps + (r > eps).astype(np.int) * r
            
            r = np.sqrt(lambd) / r
            
            if j == (startIndex - 1):
                s = 0
            else:
                s = j
                
            for i in range(s, j+1, 1):
                T = np.zeros([U.shape[1], x])
                temp1 = csr_matrix(r[:, i].reshape(x, 1)).toarray() * mask[:, i].reshape(x, 1)
                for p in range(T.shape[0]):
                    for q in range(T.shape[1]):
                        T[p, q] = U.transpose()[p, q] * temp1[q]
                        
                V[:, i] = (np.dot(np.linalg.inv(np.dot(T, U) + lambdaV * IS), np.dot(T, (Y[:, i]).reshape(x, 1)))).reshape(T.shape[0])
                
            r = abs(Y - np.dot(U, V))
            
            r = r.transpose()
            
            r = (r < eps).astype(np.int) * eps + (r > eps).astype(np.int) * r
            
            r = confidence * np.sqrt(lambd) / r
            
            if j == (startIndex - 1):
                A = []
                B = []
                for i in range(x):
                    T = np.dot(V, (np.diag(csr_matrix(r[:, i].reshape(r.shape[0], 1)).toarray() * mask[i, 0:startIndex])).reshape(r.shape[0], 1))
                
                    A.append(np.linalg.inv(np.dot(T, V.transpose()) + lambdaU * IS))
                    B.append(T * Y[i, :].reshape(Y.shape[1], 1))
                    U[i, :] = (np.dot(A[i], B[i])).reshape(1, T.shape[0])
            else:
                v = V[:, j].reshape(V.shape[0], 1)
                
                TA = A
                TB = B
                
                for i in range(x):
                    temp = np.dot(A[i], v) / forgetFactor
                    
                    if mask[i, j] == 0:
                        U[i, :] = (np.dot(TA[i], TB[i])).reshape(1, temp.shape[0])
                        continue
                    else:
                        TA[i] = A[i] / forgetFactor - r[j, i] * np.dot(temp, temp.transpose()) / (1 + r[j, i] * np.dot(v.transpose(), temp))
                        TB[i] = B[i] * forgetFactor + r[j, i] * Y[i, j] * v
                    
                    U[i, :] = (np.dot(TA[i], TB[i])).reshape(1, temp.shape[0])
                    
            r = abs(Y - np.dot(U, V))
            
            if j == (startIndex - 1):
                if ((np.sum(abs(r[:] - oldR[:]), 0) / np.sum(oldR[:])) < tol) and (c != 1) or (c > maxIter):
                    L[:, j] = (np.dot(U, V)).reshape(x)
                    break
            elif ((np.sum(abs(r[:, j] - oldR[:, j]), 0) / np.sum(oldR[:, j], 0)) < tol) or (c > maxIter):
                A = TA
                B = TB
                L[:, j] = (np.dot(U, V[:, j].reshape(V.shape[0], 1))).reshape(x)
                break
        
    return L

def nma_rescale(A, new_min, new_max):
    current_max = np.max(A[:])
    current_min = np.min(A[:])
    C =((A - current_min) * (new_max - new_min)) / (current_max - current_min) + new_min
    
    return C

def normalize(X):
    m, n = X.shape
    
    X = X - np.dot(np.ones([m, 1]), (np.mean(X, 0)).reshape(1, n))
    
    DTD = np.dot(X.transpose(), X)
    
    invTrX = np.ones([n, 1]) / (np.sqrt(np.diag(DTD))).reshape(n, 1)
    
    mul = np.dot(np.ones([m, 1]), invTrX.transpose())
    
    X = X * mul
    
    return X