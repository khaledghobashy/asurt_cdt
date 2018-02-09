# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:58:44 2018

@author: system 11
"""

import numpy as np
from base import point, vector
from pre_processor import topology_writer
from constraints import translational
from bodies_inertia import rigid

def irregularities_height(x_array,z_array,velocity,time_n):
    
    x=velocity*time_n
    z=np.interp(x,x_array,z_array)
    
    return z


def dynamic_qcm_testrig(model,chassis,wheel_mount,tire_model,road_profile):
    bodies    = model['bodies']
    joints    = model['joints']
    forces    = model['forces']
    actuators = model['actuators']
    
    ground=rigid('ground')
    origin=point('origin',[0,0,0])
    chassis_ground_joint=translational(origin,ground,chassis,vector([0,0,1]))
    
    bodies['ground']=ground
    joints['chassis_ground']=chassis_ground_joint
    
    topology_writer(bodies,joints,actuators,forces,'name')



