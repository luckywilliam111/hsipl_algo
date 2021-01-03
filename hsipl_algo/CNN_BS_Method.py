# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:00:46 2021

@author: user
"""

import numpy as np
from scipy.stats import entropy
from keras.models import Sequential
from keras.layers import Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def CNN_Entropy_Band_Selection(HIM, model, num):
    x, y, z = HIM.shape

    X = HIM.reshape(z, x, y, 1)
    
    feature_maps = model.predict(X)
    
    band_entropy = np.zeros([z])
    
    for i in range(z):
        band_entropy[i] = entropy(feature_maps[i, :])
        
    band_select_entropy = np.argsort(band_entropy * -1)
    
    band_select_entropy = band_select_entropy[:num]
    
    return band_select_entropy

def CNN_Variance_Band_Selection(HIM, model, num):
    x, y, z = HIM.shape

    X = HIM.reshape(z, x, y, 1)
    
    feature_maps = model.predict(X)
    
    band_variance = np.zeros([z])
    
    for i in range(z):
        band_variance[i] = np.var(feature_maps[i, :])
        
    band_select_variance = np.argsort(band_variance * -1)
    
    band_select_variance = band_select_variance[:num]
    
    return band_select_variance

def cnn_Featur_Model(filters, kernel, input_shape, activation, pad, maxpool, model_type='vgg', active=True, BNormalize=True, MaxPool=True, summary=True):
    model = Sequential()
    
    if model_type == 'vgg':
        for i in range(len(filters)):
            if i == 0:
                model.add(Conv2D(filters[i], (kernel[i], kernel[i]), padding=pad[i], input_shape=input_shape))
            elif i > 0:
                model.add(Conv2D(filters[i], (kernel[i], kernel[i]), padding=pad[i]))
                
            if active == True:
                model.add(Activation(activation[i]))
                
            if BNormalize == True:
                model.add(BatchNormalization())
                
            if MaxPool == True and i > 0 and (i%2 == 1):
                model.add(MaxPooling2D(pool_size=(maxpool[np.int(i/2)], maxpool[np.int(i/2)])))
                
    elif model_type == 'self':
        if len(filters) and len(kernel) and len(activation):
            for i in range(len(filters)):
                if i == 0:
                    model.add(Conv2D(filters[i], (kernel[i], kernel[i]), padding=pad[i], input_shape=input_shape))
                elif i > 0:
                    model.add(Conv2D(filters[i], (kernel[i], kernel[i]), padding=pad[i]))
                
                if active == True:
                    model.add(Activation(activation[i]))
                    
                if BNormalize == True:
                    model.add(BatchNormalization())
                    
                if MaxPool == True:
                    model.add(MaxPooling2D(pool_size=(maxpool[i], maxpool[i])))
                    
    model.add(Flatten())
    
    if summary == True:
        model.summary()
        
    return model