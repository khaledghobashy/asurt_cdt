# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:18:14 2017

@author: khale
"""

from base import orient_along_axis, vector
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
    
class triangular_prism(object):
    def __init__(self,p1,p2,p3,thickness,density=7.7):
        
        l1=(p1-p2).mag # assuming this is the base, where p3 is the vertix
        l2=(p1-p3).mag
        l3=(p2-p3).mag
        p=(l1+l2+l3)/2 # half the premiter
        
        # the normal height of the vertix from the base
        self.height=l2*np.sin(np.deg2rad(vector.angle_between(p1-p2,p1-p3)))
        # offset of vertex from base start point projected on base
        a=(p1-p3).dot(p1-p2) 
        
        self.area=np.sqrt(p*(p-l1)*(p-l2)*(p-l3))
        self.volume=self.area*thickness
        self.mass=density*self.volume*1e-3
        
        normal_vector=vector.normal(p1.a,p2.a,p3.a)
        self.cm=1/3*(p1+p2+p3)+normal_vector*thickness/2
        
        # creating a centroidal reference frame with z-axis normal to triangle
        # plane and x-axis oriented with the selected base vector
        self.C=orient_along_axis(normal_vector,p1-p2)
        
        # calculating the principle inertia properties "moment of areas" 
        Ixc=(l1*self.height**3)/36
        Iyc=((l1**3*self.height)-(l1**2*self.height*a)+(l1*self.height*a**2))/36
        Izc=((l1**3*self.height)-(l1**2*self.height*a)+(l1*self.height*a**2)+(l1*self.height**3))/36
        
        # calculating moment of inertia from the moment of area
        self.J=density*thickness*1e-3*np.diag([Ixc,Iyc,Izc])

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
        
        self.J=np.diag([Jxx,Jyy,Jzz])
        self.C=orient_along_axis(self.axis)

class composite_geometry(object):
    def __init__(self,geometries):
        
        # composite body total mass as the sum of it's subcomponents
        self.mass=sum([i.mass for i in geometries])
        # center of mass vector relative to the origin
        self.cm=vector((1/self.mass) * sum([i.mass*i.cm for i in geometries]))
        self.J=sum([i.C.dot(i.J).dot(i.C.T) + i.mass*((i.cm-self.cm).mag**2*np.eye(3)-(i.cm-self.cm).a.dot((i.cm-self.cm).a.T)) for i in geometries])
        


