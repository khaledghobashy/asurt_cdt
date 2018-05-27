# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:59:55 2017

@author: khale
"""

from base import dcm2ep, G, vector,ep2dcm
from inertia_properties import composite_geometry
import scipy as sc
import pandas as pd
import numpy as np


def principle_inertia(J):
    '''
    extracting the principle axes and the corresponding principle moment of
    inertia values from the inertia tensor calculated at the body cm aligned
    with the global frame.
    The process is done by evaluating the eigen values and eigen vectors of the
    J matrix.
    ===========================================================================
    inputs  : 
        J   : Inertia tensor
    ===========================================================================
    outputs : 
        C   : 3x3 ndarray representing the orientation of the body with the
              three principle axes
        Jp  : 3x3 diagonal ndarray storing the principle values in the diagonal
    ===========================================================================
    '''
    PJ,C=np.linalg.eig(J)
    J_Principle=np.diag(PJ)
    C[:,2]*=-1
    # C matrix transform from the body frame to the global frame
    return C, J_Principle

class rigid(object):
    
    
    def __init__(self,name,mass=1,inertia_tensor=np.eye(3),cm=vector([0,0,0]),dcm=np.eye(3)):
        '''
    A class representing the rigid body in space.
    ===========================================================================
    attributes  : 
        name    : string object represent the body name
        mass    : float representing the body mass in grams
        J       : 3x3 ndarray representing the inertia tensor at cm in gm.mm^2
        dcm     : 3x3 ndarray representing the body orientation where J is defined
        typ     : string ('floating' or 'mount') to check type of body constraints

        P       : tuple containing the evaluated euler-parameters from the dcm
        nc      : integer representing the number of constraint equations
    ===========================================================================
        '''
        self.name=name
        
        self._mass=mass
        self._J=inertia_tensor
        self._R=cm
        self._dcm=dcm
        self._P=dcm2ep(dcm)
        
        self.nc= 1
        self._geometries=pd.Series()
        self.alignment='S'
        self.notes=''
    
    
    @property
    def mass(self):
        return self._mass
    @mass.setter
    def mass(self,value):
        self._mass=value
    
    
    @property
    def R(self):
        return self._R
    @R.setter
    def R(self,value):
        self._R=value
    
    
    
    @property
    def J(self):
        return self._J
    @J.setter
    def J(self,value):
        self._J=value
    
    
    @property
    def dcm(self):
        return self._dcm
    @dcm.setter
    def dcm(self,value):
        self._dcm=value
        self._P = dcm2ep(self._dcm)
    
    @property
    def P(self):
        return self._P
    @P.setter
    def P(self,value):
        self._P=value
        self._dcm = ep2dcm(self._P)
    
    
    @property
    def geometries(self):
        return self._geometries
    @geometries.setter
    def geometries(self,value):
        self._geometries[value.name]=value
        self.update_inertia()
    
    @property
    def typ(self):
        return self._typ
    @typ.setter
    def typ(self,value):
        self._typ=value
        self.nc=(7 if value=='mount' else 1)
    
    @property    
    def m_name(self):
        if self.alignment=='S':
            return 'rbs_'+self.name[4:]
        elif self.alignment == 'R':
            return 'rbl_'+self.name[4:]
        elif self.alignment == 'L':
            return 'rbr_'+self.name[4:]
    
    def update_inertia(self):
        if len(self.geometries)==0:
            raise ValueError
        elif len(self.geometries)==1:
            geo=self.geometries[0]
            
            self.mass = geo.mass
            self.R    = geo.cm
            self.dcm  = geo.C
            self.P    = dcm2ep(self.dcm)
            self.J    = geo.J
        
        elif len(self.geometries)>1:
            geo=composite_geometry(self.geometries)
            
            self.mass = geo.mass
            self.R    = geo.cm
            self.dcm  = np.eye(3)
            self.P    = dcm2ep(self.dcm)
            self.J    = geo.J
        
    @property
    def q0(self):
        n  = self.name+'.'
        R  = self.R
        P  = self.P
        qi = pd.Series([R.x,R.y,R.z,P[0],P[1],P[2],P[3]],
                     index=[n+'x',n+'y',n+'z',n+'e0',n+'e1',n+'e2',n+'e3'])
        return qi
    
    def qd0(self,values=0):
        n  = self.name+'.'
        indices=[n+'x',n+'y',n+'z',n+'e0',n+'e1',n+'e2',n+'e3']
        if values==0:
            vel=pd.Series(np.zeros((7,)),index=indices)
        else:
            vel=pd.Series(values,index=indices)
        return vel
    
        
        
    
    @property
    def index(self):
        name=self.name
        indices=[name+'_eq%s'%i for i in range(self.nc)]
        return indices
        
    def mass_matrix(self,qi):
        P=qi[3:]
        m=self.mass*sc.sparse.eye(3)
        Gp=G(P)
        Jp=4*Gp.T.dot(self.J).dot(Gp)
        
        M=sc.sparse.bmat([[m,None],[None,Jp]])
        return M
    
    def centrifugal(self,qi,qidot):
        Pdot=qidot[3:]
        P=qi[3:].reshape((4,1))
        Gdot=G(Pdot)
        Qv=np.zeros((7,1))
        Qv[3:]=8*np.linalg.multi_dot([Gdot.T,self.J,Gdot,P])
        return Qv
    
    def inertia_force(self,qi,qidd):
        m=self.mass_matrix(qi)
        acc=qidd
        return m.dot(acc)
    
    def gravity(self):
        Qg=np.zeros((7,1))
        Qg[2,0]=-self.mass*9.81*1e3
        return Qg
    
    def equations(self,qi):
        e0,e1,e2,e3=qi[3:]
        eq=(e0**2)+(e1**2)+(e2**2)+(e3**2)-1
        return np.array([[eq]])
    
    def jac(self,qi):
        e0,e1,e2,e3=qi[3:]
        m=[0,0,0,2*e0,2*e1,2*e2,2*e3]
        return m
    
    def acc_rhs(self,qdot):
        e0,e1,e2,e3=qdot[3:]
        return np.array([[2*((e0**2)+(e1**2)+(e2**2)+(e3**2))]])
    
    
        
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name



class mount(rigid):
    def __init__(self,name):
        super().__init__(name)
        
        self.nc=7
    
    def equations(self,qi):
                
        eq1,eq2,eq3=qi[0:3]
        eq4=qi[3]-1
        eq5,eq6,eq7=qi[4:]
                
        return np.array([[eq1],[eq2],[eq3],[eq4],[eq5],[eq6],[eq7]])
    
    def jac(self,qi):
        jac=sc.sparse.csc_matrix(np.eye(7))
        return jac
    
    def acc_rhs(self,*dummy):
        return np.zeros((7,1))
    
    
    
    

