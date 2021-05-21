# -*- coding: utf-8 -*-
"""
Created on Thu May 20 00:28:25 2021

@author: WEN
"""

import numpy as np

def sym_decorrelation(W):
    K = np.dot(W, W.T)
    s, u = np.linalg.eigh(K) 
    W = np.dot((np.dot(np.dot(u, np.diag(1.0 / np.sqrt(s))), u.T)), W)
    
    return W

def g_logcosh(wx, alpha):
    return np.tanh(alpha * wx)

def gprime_logcosh(wx, alpha):
    return alpha * (1 - np.square(np.tanh(alpha * wx)))

def g_exp(wx):
    return wx * np.exp(-np.square(wx) / 2)

def gprime_exp(wx):
    return (1 - np.square(wx)) * np.exp(-np.square(wx) / 2)

class ICA:
    def __init__(self, n_components=None, fun='logcosh', alpha=1.0, maxit=200, tol=1e-04):
        self.n_components = n_components
        self.fun = fun
        self.alpha = alpha
        self.maxit = maxit
        self.tol = tol
        
    def fit(self, data):
        n,p = data.shape
        
        self.X_mean = (np.mean(data, 0)).reshape(1, p)
        
        X = data - self.X_mean
        
        X = X.T
        svd = np.linalg.svd(np.dot(X, (X.T)) / n)
        self.k = np.dot(np.diag(1 / np.sqrt(svd[1])), (svd[0].T))
        
        self.k = self.k[:self.n_components,:] 
        
        X1 = np.dot(self.k, X)
        w_init = np.random.normal(size=(self.n_components, self.n_components))
        self.W = sym_decorrelation(w_init)
        lim = 1
        it = 0
        
        if self.fun == "logcosh":
            while (lim > self.tol) and (it < self.maxit):
                wx = np.dot(self.W, X1)
                gwx = g_logcosh(wx, self.alpha)
                g_wx = gprime_logcosh(wx, self.alpha)
                
                W1 = np.dot(gwx, X1.T) / X1.shape[1] - np.dot(np.diag(g_wx.mean(axis=1)), self.W)
                W1 = sym_decorrelation(W1)
                it = it +1
                lim = np.max(np.abs(np.abs(np.diag(np.dot(W1, self.W.T))) - 1.0))
                self.W = W1
        elif self.fun == "exp":
            while (lim > self.tol) and (it < self.maxit):
                wx = np.dot(self.W, X1)
                gwx = g_exp(wx)
                g_wx = gprime_exp(wx)
                W1 = np.dot(gwx, X1.T) / X1.shape[1] - np.dot(np.diag(g_wx.mean(axis=1)), self.W)
                W1 = sym_decorrelation(W1)
                it = it +1
                lim = np.max(np.abs(np.abs(np.diag(np.dot(W1, self.W.T))) - 1.0))
                self.W = W1
                
        self.components_ = np.dot(self.W, self.k)
        
    def transform(self, data):
        X = data - self.X_mean
        
        self.mixing_ = np.linalg.pinv(self.components_)
        
        ICs = np.dot(X, self.components_.T)
        
        return ICs