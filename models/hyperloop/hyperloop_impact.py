# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:28:49 2017

@author: khale
"""

from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from constraints import spherical, revolute, universal, \
cylindrical, rotational_drive, absolute_locating,translational
from force_elements import tsda, force, tire_force
from pre_processor import topology_writer
import pandas as pd
import numpy as np
from solvers import kds, check_jacobian_dense, reactions, dds, state_space_creator
from newton_raphson import nr_kds
import matplotlib.pyplot as plt


w1 = point('w1',[0,    400,  200+5])
w2 = point('w2',[0,   -400,  200+5])
w3 = point('w3',[-800, 400,  200+5])
w4 = point('w4',[-800,-400,  200+5])

I=np.eye(3)
cm  = vector([-500,0,400+5])
dcm = I
J   = I
mass     = 130*1e3
chassis  = rigid('chassis',mass,J,cm,dcm)

wb1      = rigid('wb1',3*1e3,I,w1,I)
wb2      = rigid('wb2',3*1e3,I,w2,I)
wb3      = rigid('wb3',3*1e3,I,w3,I)
wb4      = rigid('wb4',3*1e3,I,w4,I)

axis       = vector([0,1,0])
w1_rev     = revolute(w1,wb1,chassis,axis)
w2_rev     = revolute(w2,wb2,chassis,axis)
w3_rev     = revolute(w3,wb3,chassis,axis)
w4_rev     = revolute(w4,wb4,chassis,axis)

tf1=tire_force('tvf1',wb1,250*1e6,-0.7*1e6,200,vector([0,    400,0]))
tf2=tire_force('tvf2',wb2,250*1e6,-0.7*1e6,200,vector([0,   -400,0]))
tf3=tire_force('tvf3',wb3,250*1e6,-0.7*1e6,200,vector([-800, 400,0]))
tf4=tire_force('tvf4',wb4,250*1e6,-0.7*1e6,200,vector([-800,-400,0]))
force_vector=np.array([[-20*1e6],[0],[0]])
#bf=force('brake',force_vector,chassis,cm)
bf1=force('brake1',force_vector,wb1,w1)
bf2=force('brake2',force_vector,wb2,w2)
bf3=force('brake3',force_vector,wb3,w3)
bf4=force('brake4',force_vector,wb4,w4)


wheel_drive1 = rotational_drive(w1_rev)
wheel_drive2 = rotational_drive(w2_rev)
wheel_drive3 = rotational_drive(w3_rev)
wheel_drive4 = rotational_drive(w4_rev)

wheel_drive1.pos=0
wheel_drive2.pos=0
wheel_drive3.pos=0
wheel_drive4.pos=0

bodies_list =[chassis,wb1,wb2,wb3,wb4]

joints_list =[w1_rev,w2_rev,w3_rev,w4_rev]

actuators = [wheel_drive1,wheel_drive2,wheel_drive3,wheel_drive4]
forces    = [tf1,tf2,tf3,tf4,bf1,bf2,bf3,bf4]

js=pd.Series(joints_list,index=[i.name for i in joints_list])
bs=pd.Series(bodies_list,index=[i.name for i in bodies_list])
ac=pd.Series(actuators  ,index=[i.name for i in actuators])
fs=pd.Series(forces     ,index=[i.name for i in forces])

q0   = pd.concat([i.dic    for i in bodies_list])
qd0  = pd.concat([i.qd0()  for i in bodies_list])
qd0['chassis.x']=40*1e3
qd0['wb1.x']=40*1e3
qd0['wb2.x']=40*1e3
qd0['wb3.x']=40*1e3
qd0['wb4.x']=40*1e3

topology_writer(bs,js,ac,fs,'hyperloop_file')

dynamic_hyper=dds(q0,qd0,bs,js,ac,fs,'hyperloop_file',0.5,0.001)

pos,vel,acc,react=dynamic_hyper
xaxis=np.arange(0,0.5+0.001,0.001)

pos_norms=np.linalg.norm(np.diff(pos.T),axis=0)
vel_norms=np.linalg.norm(np.diff(vel.T),axis=0)
acc_norms=np.linalg.norm(np.diff(acc.T),axis=0)


plt.figure('Chassis CG Vertical Position')
plt.plot(xaxis[:175],pos['chassis.z'][:175],label=r'$ch_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Chassis CG bounce')
plt.plot(pos['chassis.x']+500,pos['chassis.z'],label=r'$ch_{z}$')
plt.legend()
plt.xlabel('Displacement X (mm)')
plt.ylabel('Displacement Z (mm)')
plt.grid()
plt.show()

plt.figure('Chassis CG Longitudinal Position')
plt.plot(xaxis,pos['chassis.x']+500,label=r'$ch_{x}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Chassis CG Longitudinal Velocity')
plt.plot(xaxis,vel['chassis.x'],label=r'$ch_{x}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('vel (mm/sec)')
plt.grid()
plt.show()


plt.figure('Chassis CG Lateral Position')
plt.plot(xaxis,pos['chassis.y'],label=r'$ch_{y}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Wheel mount reactions')
plt.plot(xaxis,-1e-6*react['w1_rev_Fz'],label=r'$w1_{Fz}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Force (N)')
plt.grid()
plt.show()

plt.figure('Wheel Center Vertical Position')
plt.plot(xaxis[:175],pos['wb1.z'][:175],label=r'$wb1_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

