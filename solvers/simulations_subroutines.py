# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:58:44 2018

@author: system 11
"""

import numpy as np
import pandas as pd

from base import point, vector, ep2dcm
from pre_processor import topology_writer
from constraints import translational,rotational_drive
from bodies_inertia import rigid
#from solvers import dds

def irregularities_height(x_array,z_array,velocity,time_n):
    
    x=velocity*time_n
    z=np.interp(x,x_array,z_array)
    
    return z


def dynamic_qcm_testrig(model,chassis,hub_bearing,tire_model,road_profile,run_time,stepsize):
    bodies    = model['bodies']
    joints    = model['joints']
    forces    = model['forces']
    actuators = model['actuators']
    
    ground=rigid('ground')
    origin=point('origin',[0,0,0])
    chassis_ground_joint=translational(origin,ground,chassis,vector([0,0,1]))
    
    wheel_drive = rotational_drive(hub_bearing)
    wheel_drive.pos=0
    
    
    bodies['ground']=ground
    joints['chassis_ground']=chassis_ground_joint
    actuators['wheel_drive']=wheel_drive
    forces['tire_force']=tire_model
    
    topology_writer(bodies,joints,actuators,forces,'name')
    
    q0   = pd.concat([i.dic    for i in bodies])
    qd0  = pd.concat([i.qd0()  for i in bodies])
    
    dds(q0,qd0,bodies,joints,actuators,forces,'name',run_time,stepsize,road_profile)
    
    

def body_dcm(dataframe,body):
    l=[]
    for i in dataframe.T:
        p=dataframe.loc[i][body+'.e0':body+'.e3']
        dcm=ep2dcm(p)
        l.append(dcm)
    return l

def roll_angle(chassis_dcm):
    angle=[]
    for i in chassis_dcm:
        theta=np.rad2deg(np.arccos(i[:,1].dot(np.array([0,1,0]))))
        angle.append(theta)
    return angle


