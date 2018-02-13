# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:41:35 2018

@author: khale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base import point, vector
from bodies_inertia import rigid
from inertia_properties import circular_cylinder
from constraints import revolute, cylindrical, universal, spherical, absolute_locating
from solvers import kds
from pre_processor import topology_writer


A = point("A" , [0,0  ,0])
B = point("B" , [0,400,0])
C = point("C" , [1000,400,0])
D = point("D" , [1000,0  ,0])
E = point("E" , [-500,200,0])
F = point("F" , [0,200,0])
EF = point.mid_point(E,F,'EF')


I=np.eye(3)
cm=vector([0,0,0])
dcm=I
J=I
mass=1

l1g = circular_cylinder(A,B,20)
l2g = circular_cylinder(B,C,20)
l3g = circular_cylinder(C,D,20)
l4g = circular_cylinder(E,EF,10)
l5g = circular_cylinder(EF,F,20)

ground  = rigid('ground',mass,J,cm,dcm,typ='mount')
l1      = rigid('l1',l1g.mass,l1g.J,l1g.cm,l1g.C)
l2      = rigid('l2',l2g.mass,l2g.J,l2g.cm,l2g.C)
l3      = rigid('l3',l3g.mass,l3g.J,l3g.cm,l3g.C)
l4      = rigid('l4',l4g.mass,l4g.J,l4g.cm,l4g.C)
l5      = rigid('l5',l5g.mass,l5g.J,l5g.cm,l5g.C)

z=vector([0,0,1])
y=vector([1,0,0])

revA = revolute(A,l1,ground,z)
revD = revolute(D,l3,ground,z)

uniB = universal(B,l1,l2,y,-y)
uniE = universal(E,l4,ground,y,-y)
uniF = universal(F,l5,l1,y,-y)

sphC = spherical(C,l2,l3)

cylEF = cylindrical(EF,l4,l5,y)

driver= absolute_locating(l5,'x')

bodies=[ground,l1,l2,l3,l4,l5]
joints=[revA,revD,uniB,uniE,uniF,sphC,cylEF]
actuts=[driver]

bodies=pd.Series(bodies,index=[i.name for i in bodies])
joints=pd.Series(joints,index=[i.name for i in joints])
actuts=pd.Series(actuts,index=[i.name for i in actuts])


topology_writer(bodies,joints,actuts,[],'pitman_data')
q0   = pd.concat([i.dic    for i in bodies])
time=np.linspace(0,2*np.pi,200)
driver.pos_array=l5.R.x+5*np.sin(time)

sim=kds(bodies,joints,actuts,'pitman_data',time)




