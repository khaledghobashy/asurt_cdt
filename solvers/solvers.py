# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:59:46 2017

@author: khale
"""

import importlib
import pandas as pd
import numpy as np
import scipy as sc
from scipy.integrate import ode
from newton_raphson import nr_kds, nr_dds
import sys

    
def progress_bar(steps,i):
    sys.stdout.write('\r')
    length=(100*(1+i)//(4*steps))
    percentage=100*(1+i)//steps
    sys.stdout.write("Progress: ")
    sys.stdout.write("[%-25s] %d%% of %s steps." % ('='*length,percentage, steps))
    sys.stdout.flush()

def check_jacobian_sparse(sparse_jac):
    factorized=sc.sparse.linalg.splu(sparse_jac.tocsc())
    upper=factorized.U
#    lower=factorized.L
#    permC=factorized.perm_c
#    permR=factorized.perm_r
    
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

def coordinates_mapper(q):
    n=len(q)
    int2str=pd.Series(q.index,index=np.arange(0,n,1))
    str2int=pd.Series(np.arange(0,n,1),index=q.index)
    return int2str, str2int

def extract_ind(sparse_jac,q):
    mat=sparse_jac.A
    rows,cols=mat.shape
    permR=sc.linalg.lu(mat.T)[0]
    ind_cols=permR[:,rows:]
    maped=coordinates_mapper(q)[0]
    ind_coord=[maped[np.argmax(ind_cols[:,i])] for i in range(cols-rows) ]
    
    return ind_coord, ind_cols

def assign_initial_conditions(q0,qd0,qind):
    q_initial  = list(q0[qind])
    qd_initial = list(qd0[qind])
    return q_initial+qd_initial

def state_space_creator(indeces):
    
    def ssm(t,y,mass_matrix,Cq_rec,Qt,lagr):
        masses=mass_matrix.A[indeces,indeces]
        v=list(y[len(y)//2:])
        vdot=(1/masses)*(Qt[indeces]-(Cq_rec.T.dot(lagr))[indeces])
        vdot=list(vdot)
        dydt=v+vdot
        return dydt
    
    return ssm
    

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
    
    nb=len(bodies)
    
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
        
        
        lamda_df.loc[i]=lamda.reshape((7*nb,))
#        Qc_df.loc[i]=Qc.reshape((7*nb,))
#        Qa_df.loc[i]=applied.reshape((7*nb,))
#        Qv_df.loc[i]=centr.reshape((7*nb,))
#        Qi_df.loc[i]=inertia.reshape((7*nb,))
        reaction=JR(joints,q,lamda_df.loc[i])
        JR_df.loc[i]=reaction.values.reshape((len(reaction,)))
        

    return Qc_df,Qa_df,Qv_df,Qi_df,lamda_df,JR_df
        
    


def dds(q0,qd0,bodies,joints,actuators,forces,file,sim_time,stepsize):
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
    eq_f = eq_file.eq
    velf = eq_file.vel_rhs
    JR_f = eq_file.JR

    nb=len(bodies)

    # assigning initial conditions to the system NE equations
    M  = M_f(q0,bodies)
    Cq = Cq_f(q0,bodies,joints,actuators)
    Qa = Qa_f(forces,q0,qd0)
    Qv = Qv_f(bodies,q0,qd0)
    Qg = Qg_f(bodies)
    Qt = (Qa+Qv+Qg).reshape((7*nb,))
    ac = accf(q0,qd0,bodies,joints,actuators)
    
    # Initiating coordinate partitioning
    qind=extract_ind(Cq,q0)
    qstr=qind[0]
    Ids=qind[1]
    qind_index=list(coordinates_mapper(q0)[1][qstr])
    print('Independent Coordinates are: %s with indices: %s \n'%(qstr,qind_index))
    
    
    # creating dataframes to hold the simulation results at each timestep
    # with initial conditions at t0
    position_df=pd.DataFrame(columns=q0.index)
    position_df.loc[0]=q0
    
    velocity_df=pd.DataFrame(columns=q0.index)
    velocity_df.loc[0]=qd0
    
    acceleration_df=pd.DataFrame(columns=q0.index)
    
    lamdas_indices=np.concatenate([i.index for i in np.concatenate([joints,bodies,actuators])])
    lamda_df=pd.DataFrame(columns=lamdas_indices)
    
    reactions_index=np.concatenate([i.reaction_index for i in joints])
    JR_df=pd.DataFrame(columns=reactions_index)


    # assembling the coefficient matrix and the rhs vector and solving for
    # system accelerations and lagrange multipliers
    coeff_matrix=sc.sparse.bmat([[M,Cq.T],[Cq,None]],format='csc')
    b_vector=np.concatenate([Qt,ac])
    x=sc.sparse.linalg.spsolve(coeff_matrix,b_vector)
    
    # updating the dataframes with the evaluated results
    qdd0n  = x[:7*nb] # the first 7xnb elements in the x vector
    lamda0 = x[7*nb:] # the rest of elements in the x vector
    acceleration_df.loc[0]=qdd0n
    lamda_df.loc[0]=lamda0
    reaction=JR_f(joints,q0,lamda_df.loc[0])
    JR_df.loc[0]=reaction.values.reshape((len(reaction,)))

    
    # evaluating the tsda force attributes to debug
#    spring=forces[0]
#    spring_df=pd.DataFrame(columns=['deff','vel','forceS','forceD'])
#    spring_df.loc[0]=[spring.defflection,spring.velocity,spring.springforce,spring.damperforce]

    # Setting up the integrator function and the initial conditions
    ssm=state_space_creator(qind_index)
    r=ode(ssm).set_integrator('dop853')
    y0=assign_initial_conditions(q0,qd0,qstr)
    r.set_initial_value(y0).set_f_params(M,Cq,Qt,lamda0)
    
    # Setting up the time array to be used in integration steps and starting
    # the integration
    t=np.arange(0,sim_time,stepsize)
    for i,dt in enumerate(t):
        print('time_step: '+str(i))
        
        r.integrate(dt)
        print(r.y)
        
        # creating the guess vector for the vector q as the values of the 
        # previous step and the value of newly evaluated independent coordinate
        guess=position_df.loc[i]
        guess[qstr]=r.y[:len(r.y)//2]
        
        # Evaluating the dependent vector q using newton raphson
        dependent=nr_dds(eq_f,Cq_f,guess,bodies,joints,actuators,Ids)
        position_df.loc[i+1]=dependent[0]
        Cq_new=dependent[1]
        # Calculating the system velocities
        vrhs=velf(actuators)
        vind=np.array(r.y[len(r.y)//2:]).reshape((len(r.y)//2,1))
        vrhs=np.concatenate([vrhs,vind])
        velocity_df.loc[i+1]=sc.sparse.linalg.spsolve(Cq_new,vrhs)
        
        
        q=position_df.loc[i+1]
        qd=velocity_df.loc[i+1]

        # Evaluating the new coeff matrix bloks of the system NE equations
        M  = M_f(q,bodies)
        Qa = Qa_f(forces,q,qd)
        Qv = Qv_f(bodies,q,qd)
        Qg = Qg_f(bodies)
        Qt = (Qa+Qv+Qg).reshape((7*nb,))
        ac = accf(q,qd,bodies,joints,actuators)

        coeff_matrix=sc.sparse.bmat([[M,Cq.T],[Cq,None]],format='csc')
        b_vector=np.concatenate([Qt,ac])
        # Evaluating the acceleration and lagrange mult. vector
        x=sc.sparse.linalg.spsolve(coeff_matrix,b_vector)
        
        # updating the dataframes with the evaluated results
        qdd   = x[:7*nb] # the first 7xnb elements in the x vector
        lamda = x[7*nb:] # the rest of elements in the x vector
        acceleration_df.loc[i+1]=qdd
        lamda_df.loc[i+1]=lamda
#        spring_df.loc[i+1]=[spring.defflection,spring.velocity,spring.springforce,spring.damperforce]
        reaction=JR_f(joints,q,lamda_df.loc[i+1])
        JR_df.loc[i+1]=reaction.values.reshape((len(reaction,)))

        # Setting the ssm input parameters
        r.set_f_params(M,Cq_new[:M.shape[0]-len(qind_index),:],Qt,lamda)
    
    
    return position_df,velocity_df,acceleration_df,JR_df
    


    








