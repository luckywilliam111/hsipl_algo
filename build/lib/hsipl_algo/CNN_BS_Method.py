# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:00:46 2021

@author: user
"""

import numpy as np
from scipy.stats import entropy
from keras.models import Sequential
from keras.models import Model
from keras.layers import Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Dropout, Dense

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

def cnn_Featur_Model(net):
    for i in range(len(net)):
        if net[i]['layer'] == 'Input':
            net[i]['name'] = Input(shape=(net[i]['h'], net[i]['w'], net[i]['channel']))
        elif net[i]['layer'] == 'Conv2D':
            filters = net[i]['filters']
            kernel_size = net[i]['kernel_size']
            strides = (net[i]['strides'], net[i]['strides'])
            padding = net[i]['padding']
            activation = net[i]['activation']
            name = net[i]['name']
            
            net[i]['name'] = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name, activation=activation)(net[i-1]['name'])
        elif net[i]['layer'] == 'BatchNormalization':
            name = net[i]['name']
            
            net[i]['name'] = BatchNormalization(name=name)(net[i-1]['name'])
        elif net[i]['layer'] == 'MaxPooling2D':
            pool_size = net[i]['pool_size']
            strides = net[i]['strides']
            padding = net[i]['padding']
            name = net[i]['name']
            
            net[i]['name'] = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(net[i-1]['name'])
        elif net[i]['layer'] == 'Dropout':
            rate = net[i]['rate']
            name = net[i]['name']
            
            net[i]['name'] = Dropout(rate=rate, name=name)(net[i-1]['name'])
        elif net[i]['layer'] == 'Flatten':
            name = net[i]['name']
            
            net[i]['name'] = Flatten(name=name)(net[i-1]['name'])
        elif net[i]['layer'] == 'Dense':
            units = net[i]['units']
            activation = net[i]['activation']
            name = net[i]['name']
            
            net[i]['name'] = Dense(units=units, activation=activation, name=name)(net[i-1]['name'])
            
    model = Model(inputs=net[0]['name'], outputs=net[-1]['name'])
    
    model.summary()
    
    return model