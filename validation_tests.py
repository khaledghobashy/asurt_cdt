# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:18:23 2018

@author: khale
"""

import numpy as np

from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from inertia_properties import composite_geometry, triangular_prism
from constraints import spherical, revolute, universal
from pre_processor import topology_writer
import pandas as pd
from solvers import dds2

location = point('loc',[0,0,0])
endpoint = point('end',[50,0,0])

I        = np.eye(3)
ground   = rigid('ground',1,I,vector([0,0,0]),I,typ='mount')
geometry = circular_cylinder(location,endpoint,5,0)
link     = rigid('link',geometry.mass,geometry.J,geometry.cm,geometry.C)

fixation = revolute(location,ground,link,vector([0,1,0]))

bodies = [ground,link]
joints = [fixation]

js=pd.Series(joints,index=[i.name for i in joints])
bs=pd.Series(bodies,index=[i.name for i in bodies])

q0   = pd.concat([i.dic    for i in bodies])
qd0  = pd.concat([i.qd0()  for i in bodies])


topology_writer(bs,js,[],[],'validation_datafile')

run_time=0.5
stepsize=0.001
arr_size= round(run_time/stepsize)


dynamic1=dds2(q0,qd0,bs,js,[],[],'validation_datafile',run_time,stepsize)
pos,vel,acc,react=dynamic1


