# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 01:05:08 2017

@author: khale
"""

'''
Module name : geometries
'''

from base import vector, point, recursive_transformation,grf, rod2dcm
import numpy as np
import pandas as pd


class tube(object):
    '''
    A hollow tube defined by its inner and outer diameter as well as the end
    points
    outer = Outer Diameter
    inner = inner diameter
    p1 = point 1 (a point object)
    p2 = point 2 (a point object)
    body = body assigned to the tube geometry
    '''
    def __init__(self,p1,p2,outer,inner):
        
        self.name=''
        
        # defining and calculating required properties
        axis=p2-p1
        length=axis.mag
        cm_local=vector(0.5*length*axis.unit,p1.frame)
        
        # Refering the cm to the global reference frame
        # cm_global=p1+cm_local 
        cm_global=recursive_transformation(axis.frame,grf).dot(p1)+\
                    recursive_transformation(axis.frame,grf).dot(cm_local)
        
        volume=(np.pi/4)*(outer-inner)**2*length*10**-3 # cm cube
        mass=7.9*volume
        
        # defining the self.attributes
        self.cm_local  =vector(cm_local,frame=axis.frame)
        self.cm_global =point(self.name+'_cm',cm_global)
        self.length    =length
        self.volume    =volume
        self.mass      =mass
        self.p1        =p1
        self.p2        =p2
        self.r         =outer/2
        
    
    
    def __repr__(self):
        out={'name':self.name,
             'centroid':str(self.cm_global)}
        for k,v in out.items():
            print("{:<10} {:<15} ".format(k,v))




class cylinder(tube):
    '''
    A hollow tube defined by its inner and outer diameter as well as the end
    points
    outer = Outer Diameter
    inner = inner diameter
    p1 = point 1 (a point object)
    p2 = point 2 (a point object)
    body = body assigned to the tube geometry
    '''
    def __init__(self,po1,po2,body,outer=10,inner=8):
        super().__init__(po1,po2,outer,inner)
        
        self.name=body.name
                
        # updating the body-coordinate-system bcs to be coincident on the cm and
        # parallel to the grf
        body.R   = self.cm_global
        body.typ = po1.typ
    
    
    def mir(po1,po2,body,outer=10,inner=8):
        
        po1_l  ,po1_r   = po1
        po2_l  ,po2_r   = po2
        body_l  ,body_r = body
        
        left    = cylinder(po1_l,po2_l,body_l,outer,inner)
        right   = cylinder(po1_r,po2_r,body_r,outer,inner)
        
        return pd.Series([left,right],index=['l','r'])


def rod(p1,p2,body,d):
    return cylinder(p1,p2,body,d,0)


class tire(object):
    def __init__(self,center,body):
        
        self.name=body.name
        self.center=center
        self.cm_global=center
        self.axis=body.j
        
        body.R   =self.cm_global
        body.typ =center.typ
    
    def mir(center,body):
        
        center_l  ,center_r   = center
        body_l  ,body_r = body
        
        left    = tire(center_l,body_l)
        right   = tire(center_r,body_r)
        
        return pd.Series([left,right],index=['l','r'])
        


class a_arm(tube):
    '''
    an A-arm object constructed from two tubes connected at a common point (outer point).
    the center of mass of this object is calculated using the theory of composite 
    bodies (a complex body formed from simpler composite bodies with known properties).
    '''
    
    def __init__(self,fore,aft,outer,body,outer_d=10,inner_d=8):
        super().__init__(fore,aft,outer_d,inner_d)
        
        self.name=body.name
        
        self.fore = fore
        self.aft  = aft
        self.outer= outer
        
        #  generating the two tubes
        tube1=tube(fore,outer,outer_d,inner_d)
        tube2=tube(aft,outer,outer_d,inner_d)
        
        # a_arm mass as a sum of the two tubes
        self.mass=tube1.mass+tube2.mass
        
        # calculating the x,y,z coordinates of the centroid
        x=((tube1.cm_global.x*tube1.mass)+(tube2.cm_global.x*tube2.mass))/self.mass
        y=((tube1.cm_global.y*tube1.mass)+(tube2.cm_global.y*tube2.mass))/self.mass
        z=((tube1.cm_global.z*tube1.mass)+(tube2.cm_global.z*tube2.mass))/self.mass
        
        self.cm_global=point(self.name+'_cm',[x,y,z])
        
        
        #updating the body-coordinate-system bcs to be coincident on the cm and
        #parallel to the grf
        body.R   =self.cm_global
        body.typ =outer.typ
        
    
    def mir(fore,aft,outer,body,outer_d=10,inner_d=8):
        
        fore_l  ,fore_r   = fore
        aft_l   ,aft_r    = aft
        outer_l ,outer_r  = outer
        body_l  ,body_r   = body
        
        left    = a_arm(fore_l,aft_l,outer_l,body_l,outer_d,inner_d)
        right   = a_arm(fore_r,aft_r,outer_r,body_r,outer_d,inner_d)
        
        return pd.Series([left,right],index=['l','r'])
        
        
