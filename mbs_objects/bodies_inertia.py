# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:59:55 2017

@author: khale
"""

from base import dcm2ep, G, orient_along_axis
import scipy as sc
import pandas as pd
import numpy as np


def principle_inertia(J):
    PJ,C=np.linalg.eig(J)
    J_Principle=sc.sparse.diags(PJ,shape=(3,3))
    # C matrix transform from the body frame to the global I frame
    return C, J_Principle.A

class rigid(object):
    
    def __init__(self,name,mass,inertia_tensor,cm,dcm,typ='floating'):
        
        self.name=name
        self.mass=mass
        self.J=inertia_tensor
        self.R=cm
        self.dcm=dcm
        self.P=dcm2ep(dcm)
        self.typ=typ
        self.nc=(7 if typ=='mount' else 1)
        
    @property
    def dic(self):
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
    
    def qdd0(self,values=0):
        n  = self.name+'.'
        indices=[n+'x',n+'y',n+'z',n+'e0',n+'e1',n+'e2',n+'e3']
        if values==0:
            acc=pd.Series(np.zeros((7,)),index=indices)
            acc[n+'z']=9.81*1e3
        else:
            acc=pd.Series(values,index=indices)
        return acc
        
        
    
    @property
    def index(self):
        name=self.name
        indices=[name+'_eq%s'%i for i in range(self.nc)]
        return indices
        
    def mass_matrix(self,q):
        P=q[self.dic.index][3:]
        m=self.mass*sc.sparse.eye(3,format='csc')
        Gp=G(P)
        Jp=4*Gp.T.dot(self.J).dot(Gp)
        
        M=sc.sparse.bmat([[m,None],[None,Jp]],format='csc')
        return M
    
    def centrifugal(self,q,qdot):
        Pdot=qdot[self.dic.index][3:]
        P=q[self.dic.index][3:].values.reshape((4,1))
        Gdot=G(Pdot)
        Qv=np.zeros((7,1))
        Qv[3:]=8*np.linalg.multi_dot([Gdot.T,self.J,Gdot,P])
        return -Qv
    
    def inertia_force(self,q,qdd):
        m=self.mass_matrix(q)
        acc=qdd[self.dic.index]
        return m.dot(acc)
    
    def gravity(self):
        Qg=np.zeros((7,1))
        Qg[2,0]=-self.mass*9.81*1e3
        return Qg
    
    def unity_equation(self,q):
        e0,e1,e2,e3=q[self.dic.index[3:]]
        eq=(e0**2)+(e1**2)+(e2**2)+(e3**2)-1
        return np.array([[eq]])
    
    def unity_jacobian(self,q):
        e0,e1,e2,e3=q[self.dic.index[3:]]
        jac=[0,0,0,2*e0,2*e1,2*e2,2*e3]
        return jac
    
    def acc_rhs(self,qdot):
        e0,e1,e2,e3=qdot[self.dic.index[3:]]
        return np.array([[-2*((e0**2)+(e1**2)+(e2**2)+(e3**2))]])
    
    def mount_acc_rhs(self,qdot):
        e0,e1,e2,e3=qdot[self.dic.index[3:]]
        return np.concatenate([np.zeros((6,1)),np.array([[-2*((e0**2)+(e1**2)+(e2**2)+(e3**2))]])])

    
    def mount_equation(self,q):
        
        qi=q[self.dic.index]
        
        eq1,eq2,eq3=qi[0:3]
        eq4,eq5,eq6=qi[4:]
        e0,e1,e2,e3=qi[self.dic.index[3:]]
        eq7=(e0**2)+(e1**2)+(e2**2)+(e3**2)-1
        
        
        return np.array([[eq1],[eq2],[eq3],[eq4],[eq5],[eq6],[eq7]])
    
    def mount_jacobian(self,q):
        e0,e1,e2,e3=q[self.dic.index[3:]]
        b=np.array([[1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1],
                    [0,0,0,2*e0,2*e1,2*e2,2*e3]])
        jac=sc.sparse.csc_matrix(b)
        return jac
        

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
        
        self.J=sc.sparse.diags([Jxx,Jyy,Jzz]).A
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
        
        self.J=sc.sparse.diags([Jxx,Jyy,Jzz]).A
        self.C=orient_along_axis(self.axis)
      
        
    

