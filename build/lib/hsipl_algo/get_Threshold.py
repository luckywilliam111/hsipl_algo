# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:42:18 2020

@author: user
"""

import numpy as np
from skimage.filters import threshold_otsu

def graythresh_n(image, n):
    image = image.reshape((image.shape[0] * image.shape[1], 1))
    for i in range(n-1):
        thresh = threshold_otsu(image)
        bimage = image.copy()
		
        bimage[bimage < thresh] = 0
        bimage[bimage >= thresh] = 1
        
        bimage = bimage.reshape((bimage.shape[0] * bimage.shape[1], 1))

        zero_index = np.argwhere(bimage == 0)
        zero_index = list(zero_index[:, 0])
        
        image = np.delete(image, zero_index, 0)
    
    threshold = threshold_otsu(image)
        
    return threshold