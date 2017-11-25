# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:28:57 2017

@author: khale
"""


from base import point, vector, ep2dcm,rot2ep
import numpy as np
import pandas as pd
from bodies import body, mount
from constraints import revolute, spherical, universal, sph_sph, absolute_locating, rotational_drive
from pre_processor import topology_writer
from geometries import cylinder
import matplotlib.pyplot as plt
from solvers import kds
from newton_raphson import nr_kds2



A=point('A',[0,0,0])
B=point('B',[0,0,2])
C=point('C',[7.5,8.5,6.5])
D=point('D',[4,8.5,0])

ground=mount('ground')
l1=body('l1',orientation=ep2dcm(rot2ep(45,[1,1,0])))
l2=body('l2',orientation=ep2dcm([0.8794,-0.29098,-0.274,-0.2591]))
l3=body('l3',orientation=ep2dcm([0.60687,-0.36245,0.36247,-0.60684]))


l1g = cylinder(A,B,l1)
l2g = cylinder(B,C,l2)
l3g = cylinder(C,D,l3)

a_rev=revolute(A,ground,l1,vector([1,0,0]))
d_rev=revolute(D,l3,ground,vector([0,1,0]))
b_sph=spherical(B,l1,l2)
c_uni=universal(C,l2,l3,B-C,C-D)

motor = absolute_locating(l1,'y')


joints=pd.Series([a_rev,d_rev,b_sph,c_uni],index=[i.name for i in [a_rev,d_rev,b_sph,c_uni]])
bodies=pd.Series([ground,l1,l2,l3],index=[i.name for i in [ground,l1,l2,l3]])
actuators=pd.Series([motor],index=[motor.name])

topology_writer(bodies,joints,actuators,'fbooc')
from fbooc import eq, cq

time=np.linspace(0,4*np.pi,180)
#motor.set_vel(50*np.ones(len(time,)),time)
motor.set_pos(1*np.sin(1.5*time),time)
q_initial=pd.concat([i.dic for i in bodies])


fbar=kds(bodies,joints,actuators,'fbooc',time)
#motor.pos=90
#bugs=nr_kds2(eq,cq,q_initial,bodies,joints,actuators,debug=True)

plt.figure('s-v-a')
plt.plot(np.rad2deg(time),fbar[0]['l3.z'][1:])
plt.plot(np.rad2deg(time),fbar[0]['l1.z'][1:])
plt.plot(np.rad2deg(time),fbar[0]['l2.z'][1:])
plt.plot(np.rad2deg(time),fbar[1]['l3.z'][1:])
plt.grid()
plt.show()
