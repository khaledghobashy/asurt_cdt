# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:58:44 2018

@author: system 11
"""

import numpy as np

def irregularities_height(x_array,z_array,velocity,time_n):
    
    x=velocity*time_n
    z=np.interp(x,x_array,z_array)
    
    return z





