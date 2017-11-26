# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:55:29 2017

@author: khale
"""

from scipy import sparse
import numpy as np
import pandas as pd


def nr_kds(eq,jac,guess,bodies,joints,actuators,debug=False):
    '''
    eq    : system equations as a callable function of (q,joints,bodies)
    jac   : system jacobian as a callable function of (q,joints,bodies)
    guess : initial guess of the system coordinates
    '''
    q=guess.copy()
    
    
    A=jac(q,bodies,joints,actuators)
    b=-1*eq(q,bodies,joints,actuators)
    delta_q=sparse.linalg.spsolve(A,b)
    
    if debug:
        eqdf=pd.DataFrame(columns=range(len(q)))
        qdq=pd.DataFrame(columns=q.index)
    
    
    itr=0
    while np.linalg.norm(delta_q)>1e-5:
#        print('iteration: %s'%(itr))
#        print('correction norm = %s'%(np.linalg.norm(delta_q)))
#        print('equations  norm = %s \n'%(np.linalg.norm(b)))

        if debug:
            eqdf.loc[itr]=-1*b
            qdq.loc['q_'+str(itr)]=q
            qdq.loc['dq_'+str(itr)]=delta_q
        
        q=q+delta_q
        
        if itr!=0 and itr%5==0:
#            print('Recalculating Jacobian')
            A=jac(q,bodies,joints,actuators)
        b=-1*eq(q,bodies,joints,actuators)
        delta_q=sparse.linalg.spsolve(A,b)
        
        itr+=1
        


        if itr>200:
            print("Iterations exceded \n")
            break    
    
    if debug:
        return eqdf, qdq
    else:
        return q,A,itr

def nr_dds(eq,jac,guess,bodies,joints,actuators,debug=False):
    '''
    eq    : system equations as a callable function of (q,joints,bodies)
    jac   : system jacobian as a callable function of (q,joints,bodies)
    guess : initial guess of the system coordinates
    '''
    q=guess.copy()
    
    n=7*len(bodies)
    Id=np.zeros((1,n))
    Id[0,65]=1
    Ieq=np.array([0])
    
    A=jac(q,bodies,joints,actuators)
    Acon=sparse.bmat([[A],[Id]],format='csc')
    b=-1*eq(q,bodies,joints,actuators)
    bcon=np.concatenate([b,Ieq])
    delta_q=sparse.linalg.spsolve(Acon,bcon)
    
    if debug:
        eqdf=pd.DataFrame(columns=range(len(q)))
        qdq=pd.DataFrame(columns=q.index)
    
    
    itr=0
    while np.linalg.norm(delta_q)>1e-5:
#        print('iteration: %s'%(itr))
#        print('correction norm = %s'%(np.linalg.norm(delta_q)))
#        print('equations  norm = %s \n'%(np.linalg.norm(b)))

        if debug:
            eqdf.loc[itr]=-1*b
            qdq.loc['q_'+str(itr)]=q
            qdq.loc['dq_'+str(itr)]=delta_q
        
        q=q+delta_q
        
        if itr!=0 and itr%5==0:
#            print('Recalculating Jacobian')
                A=jac(q,bodies,joints,actuators)
                Acon=sparse.bmat([[A],[Id]],format='csc')

        b=-1*eq(q,bodies,joints,actuators)
        bcon=np.concatenate([b,Ieq])
        delta_q=sparse.linalg.spsolve(Acon,bcon)
        
        itr+=1
        


        if itr>50:
            print("Iterations exceded \n")
            break    
    
    if debug:
        return eqdf, qdq
    else:
        return q,Acon,itr

