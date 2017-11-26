# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:59:46 2017

@author: khale
"""

import importlib
import pandas as pd
import numpy as np
import scipy as sc
from newton_raphson import nr_kds
import sys

    
def progress_bar(steps,i):
    sys.stdout.write('\r')
    length=(100*(1+i)//(4*steps))
    percentage=100*(1+i)//steps
    sys.stdout.write("Progress: ")
    sys.stdout.write("[%-25s] %d%% of %s steps." % ('='*length,percentage, steps))
    sys.stdout.flush()

def check_jacobian(sparse_jac):
    factorized=sc.sparse.linalg.splu(sparse_jac.tocsc())
    upper=factorized.U
    lower=factorized.L
    permC=factorized.perm_c
    permR=factorized.perm_r
    
    pivots=upper.diagonal()
    ill_pv=min(abs(pivots))
    det=np.product(pivots)
    
    return det, ill_pv

def check_jacobian_dense(sparse_jac):
    permR,lower,upper=sc.linalg.lu(sparse_jac.A)
    
    pivots=upper.diagonal()
    condition=abs(pivots)>=1e-8
    ill_pvs=np.ma.masked_array(pivots,condition)
    ill_ind=np.ma.masked_array(range(len(pivots)),condition)
#    ill_index=sc.argmin(abs(pivots))
    
    return pivots, ill_pvs, ill_ind, permR
    

def kds(bodies,joints,actuators,topology_file,time_array):
    
    # importing the system equatiions from the given .py file
    eq_file=importlib.import_module(topology_file)
    cq=eq_file.cq
    eq=eq_file.eq
    vel=eq_file.vel_rhs
    acc=eq_file.acc_rhs
    
    # setting the actuatorts time history arrays
    for ac in actuators:
            
        if len(ac.pos_array)!=0:
            #print('Position Input')
            ac.vel_array=np.gradient(ac.pos_array)/np.gradient(time_array)
            ac.acc_array=np.gradient(ac.vel_array)/np.gradient(time_array)
        elif len(ac.vel_array)!=0:
            ac.pos_array=sc.integrate.cumtrapz(ac.vel_array,time_array,initial=0)
            ac.acc_array=np.gradient(ac.vel_array)/np.gradient(time_array)
    
    # checking the system jacobian for singularities and small pivots
    
    assm_config=pd.concat([i.dic for i in bodies])
    
    position_df=pd.DataFrame(columns=assm_config.index)
    position_df.loc[0]=assm_config
    
    velocity_df=pd.DataFrame(columns=assm_config.index)
    velocity_df.loc[0]=np.zeros((len(assm_config,)))
    
    acceleration_df=pd.DataFrame(columns=assm_config.index)
    acceleration_df.loc[0]=np.zeros((len(assm_config,)))
    
    convergence_df=pd.DataFrame(columns=['iteration'])
    
    print('\nRunning System Kinematic Analysis:')
    for i,step in enumerate(time_array):
        progress_bar(len(time_array),i)
        
        for ac in actuators:
            ac.pos=ac.pos_array[i]
            ac.vel=ac.vel_array[i]
            ac.acc=ac.acc_array[i]
        
        if i==0:
            g=position_df.loc[i]
        else:
            dt=time_array[i]-time_array[i-1]
            g=position_df.loc[i]+velocity_df.loc[i]*dt+acceleration_df.loc[i]*(0.5*dt**2)
#        g=position_df.loc[i]  
        position_df.loc[i+1],jac,itr=nr_kds(eq,cq,g,bodies,joints,actuators)
        velocity_df.loc[i+1]=sc.sparse.linalg.spsolve(jac,vel(actuators))
        
        convergence_df.loc[i]=itr
        
        q=position_df.loc[i+1]
        qdot=velocity_df.loc[i+1]
        acceleration_df.loc[i+1]=sc.sparse.linalg.spsolve(jac,acc(q,qdot,bodies,joints,actuators))
        
        i+=1
        
    return position_df,velocity_df, acceleration_df, convergence_df
    

def reactions(pos,vel,acc,bodies,joints,actuators,forces,file):
    
    # importing the system equatiions from the given .py file
    eq_file=importlib.import_module(file)
    Qa = eq_file.Qa
    Qv = eq_file.Qv
    Qg = eq_file.Qg
    M  = eq_file.mass_matrix
    Cq = eq_file.cq
    JR = eq_file.JR
    
    
    steps=len(pos)
    reactions_index=np.concatenate([i.reaction_index for i in joints])
    lamdas_indices=np.concatenate([i.index for i in np.concatenate([joints,bodies,actuators])])
    
    lamda_df=pd.DataFrame(columns=lamdas_indices)
    Qc_df=pd.DataFrame(columns=pos.columns)
    Qa_df=pd.DataFrame(columns=pos.columns)
    Qv_df=pd.DataFrame(columns=pos.columns)
    Qi_df=pd.DataFrame(columns=pos.columns)
    JR_df=pd.DataFrame(columns=reactions_index)
    
    
    print('\nCalculating System Reactions:')
    for i in range(steps):
        
        progress_bar(steps,i)
        
        q    = pos.loc[i]
        qdot = vel.loc[i]
        qdd  = acc.loc[i].values.reshape((len(q),1))
        
        applied=Qa(forces,q,qdot)
        centr=Qv(bodies,q,qdot)
        gravity=Qg(bodies)
        inertia=(M(q,bodies).dot(qdd))
        
        
        Qc=inertia-(applied+centr+gravity)
        lamda=sc.sparse.linalg.spsolve(Cq(q,bodies,joints,actuators).T,-Qc)
        
        
        lamda_df.loc[i]=lamda.reshape((70,))
#        Qc_df.loc[i]=Qc.reshape((70,))
#        Qa_df.loc[i]=applied.reshape((70,))
#        Qv_df.loc[i]=centr.reshape((70,))
#        Qi_df.loc[i]=inertia.reshape((70,))
        reaction=JR(joints,q,lamda_df.loc[i])
        JR_df.loc[i]=reaction.values.reshape((len(reaction,)))
        

    return Qc_df,Qa_df,Qv_df,Qi_df,lamda_df,JR_df
        
    


def dds(q0,qd0,qdd0,bodies,joints,actuators,forces,file,sim_time):
    '''
    Dynamically Driven Systems Solver
    '''
    # importing the system equatiions from the given .py file
    eq_file=importlib.import_module(file)
    Qa_f = eq_file.Qa
    Qv_f = eq_file.Qv
    Qg_f = eq_file.Qg
    M_f  = eq_file.mass_matrix
    Cq_f = eq_file.cq
    accf = eq_file.acc_rhs
    
    M  = M_f(q0,bodies)
    Cq = Cq_f(q0,bodies,joints,actuators)
    Qa = Qa_f(forces,q0,qd0)
    Qv = Qv_f(bodies,q0,qd0)
    Qg = Qg_f(bodies)
    Qt = Qa+Qv+Qg
    ac = accf(q0,qd0,bodies,joints,actuators)
    
    nb=len(bodies)
    
    coeff_matrix=sc.sparse.bmat([[M,Cq.T],[Cq,None]],format='csc')
    b_vector=np.concatenate([Qt.reshape((70,)),ac])
    
    x=sc.sparse.linalg.spsolve(coeff_matrix,b_vector)
    
    return x
    











