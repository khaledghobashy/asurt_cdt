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
from solvers import kds, reactions
from pre_processor import topology_writer


mount_1   = point("mount_1" , [-4500  , 286  ,54])
mount_2   = point("mount_2" , [-4500  ,-286  ,54])
coupler_1 = point("C1"      , [-4376  , 364  ,50])
coupler_2 = point("C2"      , [-4376  ,-364  ,50])
E = point("E" , [-4320  , 608  ,157])
F = point("F" , [-4349  ,-285  ,85])
EF = point.mid_point(E,F,'EF')


I=np.eye(3)
cm=vector([0,0,0])
dcm=I
J=I
mass=1

l1g = circular_cylinder(mount_1,coupler_1,40)
l2g = circular_cylinder(coupler_1,coupler_2,40)
l3g = circular_cylinder(coupler_2,mount_2,40)
l4g = circular_cylinder(E,EF,40)
l5g = circular_cylinder(EF,F,70,40)

ground  = rigid('ground',mass,J,cm,dcm,typ='mount')
l1      = rigid('l1',l1g.mass,l1g.J,l1g.cm,l1g.C)
l2      = rigid('l2',l2g.mass,l2g.J,l2g.cm,l2g.C)
l3      = rigid('l3',l3g.mass,l3g.J,l3g.cm,l3g.C)
l4      = rigid('l4',l4g.mass,l4g.J,l4g.cm,l4g.C)
l5      = rigid('l5',l5g.mass,l5g.J,l5g.cm,l5g.C)

z=vector([0,0,1])
y=vector([0,1,0])

revA = revolute(mount_1,l1,ground,z)
revD = revolute(mount_2,l3,ground,z)

uniB = universal(coupler_1,l1,l2,y,-y)
uniE = universal(E,l4,ground,y,-y)
uniF = universal(F,l5,l3,y,-y)

sphC = spherical(coupler_2,l2,l3)

cylEF = cylindrical(EF,l4,l5,y)

driver= absolute_locating(l5,'y')

bodies=[ground,l1,l2,l3,l4,l5]
joints=[revA,revD,uniB,uniE,uniF,sphC,cylEF]
actuts=[driver]

bodies=pd.Series(bodies,index=[i.name for i in bodies])
joints=pd.Series(joints,index=[i.name for i in joints])
actuts=pd.Series(actuts,index=[i.name for i in actuts])


topology_writer(bodies,joints,actuts,[],'pitman_data')
q0   = pd.concat([i.dic    for i in bodies])

time=np.linspace(0,2*np.pi,200)
driver.pos_array=l5.R.y+80*np.sin(5*time)

sim=kds(bodies,joints,actuts,'pitman_data',time)
sim_reactions=reactions(sim[0],sim[1],sim[2],bodies,joints,actuts,[],'pitman_data')[5]



plt.figure('ss')
plt.plot(sim[0]['l5.y'][1:]-l5.R.y,sim[0]['l2.y'][1:],label='coupler_movement')
plt.xlabel('Displacement (mm)')
plt.ylabel('Displacement (mm)')
plt.legend()
plt.grid()
plt.show()

plt.figure('react')
plt.plot(sim[0]['l5.y'][1:]-l5.R.y,sim_reactions['E_uni_Fy'][1:]*1e-6,label='actuator_fixation_Fy')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid()
plt.show()



