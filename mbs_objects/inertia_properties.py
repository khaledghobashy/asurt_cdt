# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:18:14 2017

@author: khale
"""

from base import orient_along_axis
import scipy as sc
import numpy as np

class thin_rod(object):
    def __init__(self,p1,p2,mass):
        self.p1=p1
        self.p2=p2
        self.mass=mass
        
        self.axis=p2-p1
        self.l=self.axis.mag
        
        self.cm=p1+0.5*self.axis
        Jxx=Jyy=(self.mass/12)*self.l**2
        Jzz=0
        
        self.J=sc.sparse.diags([Jxx,Jyy,Jzz])
        self.C=orient_along_axis(self.axis)
        
class rectangle_prism(object):
    def __init__(self,l,w,h,mass):
        pass

class circular_cylinder(object):
    def __init__(self,p1,p2,do,di=0):
        self.p1=p1
        self.p2=p2
        
        self.axis=p2-p1
        self.l=self.axis.mag
        
        self.mass=7.7*np.pi*(do**2-di**2)*self.l*1e-3

        
        self.cm=p1+0.5*self.axis
        Jxx=Jyy=(self.mass/12)*(3*do**2+3*di**2+self.l**2)
        Jzz=(self.mass/2)*(do**2+di**2)
        
        self.J=sc.sparse.diags([Jxx,Jyy,Jzz])
        self.C=orient_along_axis(self.axis)



