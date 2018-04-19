# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:58:44 2018

@author: system 11
"""

#import numpy as np
#import pandas as pd
#
#from base import point, vector, ep2dcm
#from pre_processor import topology_writer
#from constraints import translational,rotational_drive
#from bodies_inertia import rigid
#from solvers import dds
#
#    
#
#def body_dcm(dataframe,body):
#    l=[]
#    for i in dataframe.T:
#        p=dataframe.loc[i][body+'.e0':body+'.e3']
#        dcm=ep2dcm(p)
#        l.append(dcm)
#    return l
#
#def roll_angle(chassis_dcm):
#    angle=[]
#    for i in chassis_dcm:
#        theta=np.rad2deg(np.arccos(i[:,1].dot(np.array([0,1,0]))))
#        angle.append(theta)
#    return angle
#

