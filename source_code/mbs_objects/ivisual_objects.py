# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:25:04 2018

@author: khaled.ghobashy
"""

import ivisual as v
import numpy as np
from base import vector, rod2dcm
from geometries import  a_arm, cylinder, tire
import time

def T(coord):
    '''
    Transfrom the MBS coordinates to the visual python coordinate system
    coord: list of x,y,z coordinates
    '''
    if isinstance(coord,vector):
        new_pos=(coord[1,0],coord[2,0],coord[0,0])
    else:
        new_pos=(coord[1],coord[2],coord[0])
    return np.array(new_pos)


class frame(object):
    def __init__(self,frame,length=100):
        i=T(frame[:,0])
        j=T(frame[:,1])
        k=T(frame[:,2])
        origin=v.vector(T(np.resize(frame.loc,(3,))))
        
        self.i=v.arrow(pos=origin,axis=v.vector(i),color=v.color.red,length=length)
        self.j=v.arrow(pos=origin,axis=v.vector(j),color=v.color.blue,length=length)
        self.k=v.arrow(pos=origin,axis=v.vector(k),color=v.color.green,length=length)

class sph(object):
    '''
    point : Point object
    c     : Color
    r     : radius
    '''
    def __init__(self,point,c=v.color.blue,r=10):
        #scene=canvas(title='Ref',background=color.gray(0.2))
        self.p=point
        self.shape=v.sphere(pos=v.vector(T(self.p)), radius=r,color=c)
    
    def animate(self,data,ind):
        p_name=self.p.name+'.'
        p =T(data.T[ind][p_name+'x':p_name+'z'])
        self.shape.pos=v.vector(p)
        


class arm(object):
    def __init__(self,arm_goe,c=v.color.red):
        #scene=canvas(title='Ref',background=color.gray(0.2))
        self.fore  =arm_goe.fore
        self.aft   =arm_goe.aft
        self.outer =arm_goe.outer
        self.r=r   =arm_goe.r
                
        
        self.c1=v.cylinder(pos=v.vector(T(self.fore)),axis=v.vector(T(self.outer-self.fore)), radius=r,color=c)
        self.c2=v.cylinder(pos=v.vector(T(self.aft)) ,axis=v.vector(T(self.outer-self.aft)) , radius=r,color=c)
        self.c3=v.cylinder(pos=v.vector(T(self.fore)),axis=v.vector(T(self.aft-self.fore))  , radius=r,color=c)

    
    
    def animate(self,data,ind):
        f_name=self.fore.name+'.'
        r_name=self.aft.name+'.'
        o_name=self.outer.name+'.'
        
        front =T(data.T[ind][f_name+'x':f_name+'z'])
        rear  =T(data.T[ind][r_name+'x':r_name+'z'])
        outer =T(data.T[ind][o_name+'x':o_name+'z'])
        
        self.c1.pos=v.vector(front)
        self.c1.axis=v.vector(outer-front)
        
        self.c2.pos=v.vector(rear)
        self.c2.axis=v.vector(outer-rear)
        
        self.c3.pos=v.vector(front)
        self.c3.axis=v.vector(rear-front)



class link(object):
    def __init__(self,tube,c=v.color.white):
        self.p1=tube.p1
        self.p2=tube.p2
        self.r=r=tube.r
        self.c=0
       
        self.cyl=v.cylinder(pos=v.vector(T(self.p1)),axis=v.vector(T(self.p2-self.p1)), radius=r,color=c)

    def animate(self,data,ind):
        p1_name=self.p1.name+'.'
        p2_name=self.p2.name+'.'
        
        p1 =T(data.T[ind][p1_name+'x':p1_name+'z'])
        p2 =T(data.T[ind][p2_name+'x':p2_name+'z'])
        
        self.cyl.pos=v.vector(p1)
        self.cyl.axis=v.vector(p2-p1)

class tire_geo(object):
    def __init__(self,wheel_geo,c=v.color.gray(0.1)):
        self.name=wheel_geo.name
        self.center=wheel_geo.center
        self.axis=wheel_geo.axis
        self.r=508/2
        self.w=210
        
        spin_ax=105*self.axis if self.center.y<0 else -105*self.axis
        self.cyl=v.cylinder(pos=v.vector(T(self.center)),axis=v.vector(T(spin_ax)), radius=self.r,color=c)
    
    def animate(self,data,ind):
        p1_name=self.name+'.'
        b_name=self.name+'.'
        
        p1 =T(data.T[ind][p1_name+'x':p1_name+'z'])
        ax =rod2dcm(vector(data.T[ind][b_name+'y1':b_name+'y3']))[:,1]
        
        spin_ax=105*ax if self.center.y<0 else -105*ax
        self.cyl.pos=v.vector(p1)
        self.cyl.axis=v.vector(T(spin_ax))



def assign_geo(system_geometries):
    geo_list=[]
    for geo in system_geometries:
        if type(geo)==cylinder:
            geo_list.append(link(geo))
        elif type(geo)==a_arm:
            geo_list.append(arm(geo))
        elif type(geo)==tire:
            geo_list.append(tire_geo(geo))
    return geo_list


def animate(q,p,geometries,t=0.005,l=10):
    for j in range(l):
        for i in range(len(p)):
            time.sleep(t)
            for g in geometries:
                if isinstance(g,tire_geo):
                    g.animate(q,i)
                else:
                    g.animate(p,i)