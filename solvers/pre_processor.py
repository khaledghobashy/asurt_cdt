# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 07:58:27 2017

@author: khale
"""

from bodies import mount
from force_elements import tsda
from constraints import absolute_locating
import numpy as np

def topology_writer(bodies,joints,actuators,forces,file_name):
    columns =[i.name for i in bodies ]
    rows    =[i.name for i in joints ]+columns+[i.name for i in actuators]
    joints_eq_ind=np.concatenate([i.reaction_index for i in joints])
    n=7*len(bodies)
    file=open(file_name+'.py','w')
    file.flush()
    
    file.write("import numpy as np \n")
    file.write("import pandas as pd \n")
    file.write("from scipy import sparse \n")
    
    file.write("\n")
    file.write("\n")
    
    # Assembling the system jacobian based on the given joints, bodies
    # and actuators
    file.write("jac_df=pd.DataFrame([%s *[None]],columns=%s,index=%s)\n"%(len(bodies),columns,rows))
    file.write("def cq(q,bodies,joints,actuators): \n")
    for j in joints:
        name=j.name
        ibody=j.i_body.name
        jbody=j.j_body.name
        file.write("\t jac_df.loc['%s','%s']=joints['%s'].jacobian_i(q)\n"%(name,ibody,name))
        file.write("\t jac_df.loc['%s','%s']=joints['%s'].jacobian_j(q)\n"%(name,jbody,name))
    
    
    for b in bodies:
        if b.typ=='mount':
            file.write("\t jac_df.loc['%s','%s']=bodies['%s'].mount_jacobian(q)\n"%(b.name,b.name,b.name))
        else:
            file.write("\t jac_df.loc['%s','%s']=bodies['%s'].unity_jacobian(q)\n"%(b.name,b.name,b.name))

    for a in actuators:
        if isinstance(a,absolute_locating):
            name=a.name
            body=a.body.name
            file.write("\t jac_df.loc['%s','%s']=actuators['%s'].jacobian()\n"%(name,body,name))
        else:
            name=a.name
            ibody=a.i_body.name
            jbody=a.j_body.name
            file.write("\t jac_df.loc['%s','%s']=actuators['%s'].jacobian_i(q)\n"%(name,ibody,name))
            file.write("\t jac_df.loc['%s','%s']=actuators['%s'].jacobian_j(q)\n"%(name,jbody,name))

            
    
    file.write("\t jac=sparse.bmat(jac_df,format='csc') \n")
    file.write("\t return jac \n")
    
    file.write("\n")
    file.write("\n")
    
    ######################################################################    
    # Assembling the system constraint equations vector based on the 
    # given joints, bodies and actuators
    file.write("eq_s=pd.Series([%s *[None]],index=%s)\n"%(len(rows),rows))
    file.write("def eq(q,bodies,joints,actuators): \n")
    njc=0
    for j in joints:
        njc=njc+j.nc
        file.write("\t eq_s['%s']=joints['%s'].equations(q)\n"%(j.name,j.name))

    nbc=0
    for b in bodies:
        nbc=nbc+1
        if b.typ=='mount':
            nbc=nbc+6
            file.write("\t eq_s['%s']=bodies['%s'].mount_equation(q)\n"%(b.name,b.name))
        else:
            file.write("\t eq_s['%s']=bodies['%s'].unity_equation(q)\n"%(b.name,b.name))
    
    nac=0
    for a in actuators:
        nac=nac+1
        file.write("\t eq_s['%s']=actuators['%s'].equations(q)\n"%(a.name,a.name))

    file.write("\t system=sparse.bmat(eq_s.values.reshape((%s,1)),format='csc') \n"%(len(rows)))
    file.write("\t return system.A.reshape((%s,)) \n"%str(njc+nac+nbc))

    file.write("\n")
    file.write("\n")
    
    ####################################################################
    # Assembling the system constraint rhs velocity vector
    file.write("vel_rhs_s=pd.Series([%s *[None]],index=%s)\n"%(len(rows),rows))
    file.write("def vel_rhs(actuators): \n")
    file.write("\t vrhs=np.zeros((%s,1))\n"%str(njc+nbc))
    for a in actuators:
        file.write("\t vrhs=np.concatenate([vrhs,actuators['%s'].vel_rhs()]) \n"%(a.name))
    file.write("\t return vrhs \n")
    
    ######################################################################    
    # Assembling the system accelertaion rhs equations vector based on the 
    # given joints, bodies and actuators
    file.write("acc_rhs_s=pd.Series([%s *[None]],index=%s)\n"%(len(rows),rows))
    file.write("def acc_rhs(q,qdot,bodies,joints,actuators): \n")
    for j in joints:
        file.write("\t acc_rhs_s['%s']=joints['%s'].acc_rhs(q,qdot)\n"%(j.name,j.name))

    for b in bodies:
        if b.typ=='mount':
            file.write("\t acc_rhs_s['%s']=bodies['%s'].mount_acc_rhs(qdot)\n"%(b.name,b.name))
        else:
            file.write("\t acc_rhs_s['%s']=bodies['%s'].acc_rhs(qdot)\n"%(b.name,b.name))
    
    for a in actuators:
        file.write("\t acc_rhs_s['%s']=actuators['%s'].acc_rhs(q,qdot)\n"%(a.name,a.name))

    file.write("\t system=sparse.bmat(acc_rhs_s.values.reshape((%s,1)),format='csc') \n"%(len(rows)))
    file.write("\t return system.A.reshape((%s,)) \n"%str(njc+nac+nbc))

    file.write("\n")
    file.write("\n")
    
    ######################################################################    
    # Assembling the system mass matrix 
    file.write("mass_matrix_df=pd.DataFrame([%s *[None]],columns=%s,index=%s)\n"%(len(columns),columns,columns))
    file.write("def mass_matrix(q,bodies): \n")
    for b in bodies:
        file.write("\t mass_matrix_df.loc['%s','%s']=bodies['%s'].mass_matrix(q)\n"%(b.name,b.name,b.name))
    
    file.write("\t mass_matrix=sparse.bmat(mass_matrix_df,format='csc') \n")
    file.write("\t return mass_matrix \n")

    file.write("\n")
    file.write("\n")
    
    ######################################################################    
    # Assembling the gravitational force vector
    file.write("Qg_s=pd.Series([%s *np.zeros((7,1))],index=%s)\n"%(len(columns),columns))
    file.write("def Qg(bodies): \n")
    for b in bodies:
        file.write("\t Qg_s['%s']=bodies['%s'].gravity()\n"%(b.name,b.name))

    file.write("\t system=sparse.bmat(Qg_s.values.reshape((%s,1)),format='csc') \n"%(len(columns)))
    file.write("\t return system.A.reshape((%s,)) \n" %n)

    file.write("\n")
    file.write("\n")
    
    
    ######################################################################    
    # Assembling the centrifugal force vector
    file.write("Qv_s=pd.Series([%s *np.zeros((7,1))],index=%s)\n"%(len(columns),columns))
    file.write("def Qv(bodies,q,qdot): \n")
    for b in bodies:
        file.write("\t Qv_s['%s']=bodies['%s'].centrifugal(q,qdot)\n"%(b.name,b.name))

    file.write("\t system=sparse.bmat(Qv_s.values.reshape((%s,1)),format='csc') \n"%(len(columns)))
    file.write("\t return system.A.reshape((%s,)) \n" %n)

    file.write("\n")
    file.write("\n")
    
    
    ######################################################################    
    # Assembling the external force vector
    file.write("Qa_s=pd.Series([%s *np.zeros((7,1))],index=%s)\n"%(len(columns),columns))
    file.write("def Qa(forces,q,qdot): \n")

    for f in forces:
        if isinstance(f,tsda):
            file.write("\t Qi,Qj=forces['%s'].equation(q,qdot) \n" %f.name)
            file.write("\t Qa_s['%s']=Qi\n"%(f.bodyi.name))
            file.write("\t Qa_s['%s']=Qj\n"%(f.bodyj.name))
        else:
            file.write("\t Qa_s['%s']=forces['%s'].equation(q,qdot)\n" %(f.bodyi.name,f.name))
            

    file.write("\t system=sparse.bmat(Qa_s.values.reshape((%s,1)),format='csc') \n"%(len(columns)))
    file.write("\t return system.A.reshape((%s,)) \n" %n)

    file.write("\n")
    file.write("\n")
    
    ######################################################################    
    # Assembling the joint reactions vector
    file.write("JR_s=pd.Series(np.zeros((%s)),index=%s)\n"%(6*len(joints),list(joints_eq_ind)))
    file.write("def JR(joints,q,lamda): \n")

    for j in joints:
#        file.write("\t print(joints['%s'].reactions(q,lamda))\n"%(j.name))
        file.write("\t JR_s[%s]=joints['%s'].reactions(q,lamda)\n"%(j.reaction_index,j.name))

#    file.write("\t system=sparse.bmat(JR_s.values.reshape((%s,1)),format='csc') \n"%(len(columns)))
    file.write("\t return JR_s \n")

    file.write("\n")
    file.write("\n")
    
    file.close()
    
###############################################################################
###############################################################################
###############################################################################

def mirror(model,sym='l'):
    
    points=model['points']
    points_right = points.copy()
    points_left  = points.copy()
    points_mir_mul=np.array(len(points)*[1,-1,1])

    if sym not in 'rl':
        raise ValueError('Un-authorized type')
    elif sym=='r':
        r='_rt'
        points_right.index+=r
        points_left*=points_mir_mul
    elif sym=='l':
        l='_lf'
        points_left.index+=l
        points_right*=points_mir_mul
        
    return points_right, points_left
        
    
    
    
    






