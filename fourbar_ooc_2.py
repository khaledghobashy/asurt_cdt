# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:28:57 2017

@author: khale
"""


from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from constraints import spherical, revolute, universal, \
cylindrical, rotational_drive, absolute_locating,translational
from force_elements import tsda, force, moment
from pre_processor import topology_writer
import pandas as pd
import numpy as np
from solvers import kds, check_jacobian_dense, reactions, dds, state_space_creator
from newton_raphson import nr_kds
import matplotlib.pyplot as plt



A=point('A',[0,0,0])
B=point('B',[0,0,20])
C=point('C',[75,85,65])
D=point('D',[40,85,0])

##############################################################################
I=np.eye(3)
cm=vector([0,0,0])
dcm=I
J=I
mass=1
ground  = rigid('ground',mass,J,cm,dcm,typ='mount')
##############################################################################
l1_g  = circular_cylinder(A,B,15)
cm    = l1_g.cm
dcm   = l1_g.C
J     = l1_g.J
mass  = l1_g.mass
l1    = rigid('l1',mass,J,cm,dcm)
##############################################################################
l2_g  = circular_cylinder(B,C,15)
cm    = l2_g.cm
dcm   = l2_g.C
J     = l2_g.J
mass  = l2_g.mass 
l2    = rigid('l2',mass,J,cm,dcm)
##############################################################################
l3_g  = circular_cylinder(C,D,15)
cm    = l3_g.cm
dcm   = l3_g.C
J     = l3_g.J
mass  = l3_g.mass 
l3    = rigid('l3',mass,J,cm,dcm)
##############################################################################


a_rev=revolute(A,ground,l1,vector([1,0,0]))
d_rev=revolute(D,l3,ground,vector([0,1,0]))
b_uni=universal(B,l1,l2,A-B,B-C)
c_sph=spherical(C,l2,l3)

moment_vector=np.array([[-10*1e9],[0],[0]])
vf=moment('vertical_force',moment_vector,l1,vector([0,0,0]))


joints= pd.Series([a_rev,d_rev,b_uni,c_sph],index=[i.name for i in [a_rev,d_rev,b_uni,c_sph]])
bodies= pd.Series([ground,l1,l2,l3],index=[i.name for i in [ground,l1,l2,l3]])
fc    = pd.Series([vf],index=[i.name for i in [vf]])
q_0   = pd.concat([i.dic for i in bodies])
qd_0  = pd.concat([i.qd0()  for i in bodies])
qdd_0 = pd.concat([i.qdd0() for i in bodies])

fc=[]
topology_writer(bodies,joints,[],fc,'fbar_dyn')

dynamic1_fbar=dds(q_0,qd_0,qdd_0,bodies,joints,[],fc,'fbar_dyn',1,1/50)
posf,velf,accf,reactf=dynamic1_fbar




