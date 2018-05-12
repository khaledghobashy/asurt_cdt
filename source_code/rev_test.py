# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:04:15 2018

@author: khale
"""

import numpy as np
import pandas as pd
from bodies_inertia import rigid
from base import point, vector
from constraints import revolute, rotational_actuator
from inertia_properties import circular_cylinder
from pre_processor import topology_writer
from solvers import kds

O = point('O',[0,0,0])
A = point('A',[2000,0,0])

ground = rigid('ground',typ='mount')
link   = rigid('link')
link_geo = circular_cylinder('l',link,O,A,20)
link.update_inertia()

y = vector([0,1,0])

rev = revolute('rev',O,ground,link,y)
rot = rotational_actuator('rot',rev)
rot.pos_f = lambda t : -np.pi*np.sin(t)

bs = pd.Series([ground,link],index=['ground','link'])
js = pd.Series([rev],index=['rev'])
ac = pd.Series([rot],index=['rot'])

q0 = pd.concat([i.dic    for i in bs])

topology_writer(bs,js,ac,[],'sd')

t = np.linspace(0,10,100)
soln = kds(bs,js,ac,'sd',t)



