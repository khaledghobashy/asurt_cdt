# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:24:02 2017

@author: khale
"""


from base import reference_frame, vector, point, dcm2ep, grf, ep2dcm
import numpy as np
import pandas as pd
from scipy import sparse


class body(reference_frame):
    '''
    A class representing rigid bodies in three dimensional space as a reference
    frame where the orientation is represented by the three rodriguez parameters
    extracted from the body dcm/orientation matrix
    The class is a subclass of the reference frame class.
    No inertia properties yet.
    '''
    def __init__(self,name,orientation=np.eye(3),location=np.zeros((3,1)),parent=grf):
        super().__init__(name,orientation,location,parent)
          
        self._ep=dcm2ep(orientation)
        self.typ=None
    
    @property
    def R(self):
        '''Absolute position vector of body center of gravity'''
        return vector(self.loc)
    
    @R.setter
    def R(self,value):
        loc_type   = isinstance(value,(list,tuple,np.ndarray,point,vector,pd.Series))
        if not loc_type : raise TypeError('should be a list, tuple or ndarray')
        loc_length = len(value)==3
        if not loc_length : raise ValueError('location should be a vector of 3 components')
        self.loc=vector(value)
    
    @property
    def ep(self):
        return self._ep
    @ep.setter
    def ep(self,value):
        self._ep=value
        self.dcm=ep2dcm(value)
    
    @property
    def dic(self):
        n=self.name+'.'
        R  =self.R
        ep =self.ep
        qi=pd.Series([R.x,R.y,R.z,ep[0],ep[1],ep[2],ep[3]],
                     index=[n+'x',n+'y',n+'z',n+'e0',n+'e1',n+'e2',n+'e3'])
        return qi
    
    def unity_equation(self,q):
        e0,e1,e2,e3=q[list(self.dic.index)[3:]]
        eq=(e0**2)+(e1**2)+(e2**2)+(e3**2)-1
        return np.array([[eq]])
    
    def unity_jacobian(self,q):
        e0,e1,e2,e3=q[list(self.dic.index)[3:]]
        jac=[0,0,0,2*e0,2*e1,2*e2,2*e3]
        return jac
    
    def acc_rhs(self,qdot):
        e0,e1,e2,e3=qdot[list(self.dic.index)[3:]]
        return np.array([[-2*((e0**2)+(e1**2)+(e2**2)+(e3**2))]])
        
    def mir(name):
        right = body(name+'.r')
        left  = body(name+'.l')
        return pd.Series([left,right],index=['l','r'])
    
    def __repr__(self):
        return 'body: '+self.name


class mount(body):
    def __init__(self,name):
        super().__init__(name)
        
    def mount_equation(self,q):
        
        qi=q[list(self.dic.index)]
        
        eq1,eq2,eq3=qi[0:3]
        eq4,eq5,eq6=qi[4:]
        e0,e1,e2,e3=qi[list(self.dic.index)[3:]]
        eq7=(e0**2)+(e1**2)+(e2**2)+(e3**2)-1
        
        
        return np.array([[eq1],[eq2],[eq3],[eq4],[eq5],[eq6],[eq7]])
    
    def mount_jacobian(self,q):
        e0,e1,e2,e3=q[list(self.dic.index)[3:]]
        b=np.array([[1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1],
                    [0,0,0,2*e0,2*e1,2*e2,2*e3]])
        jac=sparse.csr_matrix(b)
        return jac
    
    def acc_rhs(self,qdot):
        e0,e1,e2,e3=qdot[list(self.dic.index)[3:]]
        return np.concatenate([np.zeros((6,1)),np.array([[-2*((e0**2)+(e1**2)+(e2**2)+(e3**2))]])])
    
    def mir(name):
        right = mount(name+'.r')
        left  = mount(name+'.l')
        return pd.Series([left,right],index=['l','r'])
    