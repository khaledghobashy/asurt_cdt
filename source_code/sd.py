import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([2 *[None]],columns=['ground', 'link'],index=['rev', 'ground', 'link', 'rot'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['rev','ground']=joints['rev'].jacobian_i(q)
	 jac_df.loc['rev','link']=joints['rev'].jacobian_j(q)
	 jac_df.loc['ground','ground']=bodies['ground'].mount_jacobian(q)
	 jac_df.loc['link','link']=bodies['link'].unity_jacobian(q)
	 jac_df.loc['rot','ground']=actuators['rot'].jacobian_i(q)
	 jac_df.loc['rot','link']=actuators['rot'].jacobian_j(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([4 *[None]],index=['rev', 'ground', 'link', 'rot'])
def eq(q,bodies,joints,actuators): 
	 eq_s['rev']=joints['rev'].equations(q)
	 eq_s['ground']=bodies['ground'].mount_equation(q)
	 eq_s['link']=bodies['link'].unity_equation(q)
	 eq_s['rot']=actuators['rot'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((14,)) 


vel_rhs_s=pd.Series([4 *[None]],index=['rev', 'ground', 'link', 'rot'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((13,1))
	 vrhs=np.concatenate([vrhs,actuators['rot'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([4 *[None]],index=['rev', 'ground', 'link', 'rot'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['rev']=joints['rev'].acc_rhs(q,qdot)
	 acc_rhs_s['ground']=bodies['ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['link']=bodies['link'].acc_rhs(qdot)
	 acc_rhs_s['rot']=actuators['rot'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((14,)) 


mass_matrix_df=pd.DataFrame([2 *[None]],columns=['ground', 'link'],index=['ground', 'link'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['ground','ground']=bodies['ground'].mass_matrix(q)
	 mass_matrix_df.loc['link','link']=bodies['link'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([2 *np.zeros((7,1))],index=['ground', 'link'])
def Qg(bodies): 
	 Qg_s['ground']=bodies['ground'].gravity()
	 Qg_s['link']=bodies['link'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((2,1)),format='csc') 
	 return system.A.reshape((14,)) 


Qv_s=pd.Series([2 *np.zeros((7,1))],index=['ground', 'link'])
def Qv(bodies,q,qdot): 
	 Qv_s['ground']=bodies['ground'].centrifugal(q,qdot)
	 Qv_s['link']=bodies['link'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((2,1)),format='csc') 
	 return system.A.reshape((14,)) 


Qa_s=pd.Series([2 *np.zeros((7,1))],index=['ground', 'link'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((2,1)),format='csc') 
	 return system.A.reshape((14,)) 


JR_s=pd.Series(np.zeros((6)),index=['rev_Fx', 'rev_Fy', 'rev_Fz', 'rev_Mx', 'rev_My', 'rev_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['rev_Fx', 'rev_Fy', 'rev_Fz', 'rev_Mx', 'rev_My', 'rev_Mz']]=joints['rev'].reactions(q,lamda)
	 return JR_s 


