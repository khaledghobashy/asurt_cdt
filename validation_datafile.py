import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([2 *[None]],columns=['ground', 'link'],index=['loc_rev', 'ground', 'link'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['loc_rev','ground']=joints['loc_rev'].jacobian_i(q)
	 jac_df.loc['loc_rev','link']=joints['loc_rev'].jacobian_j(q)
	 jac_df.loc['ground','ground']=bodies['ground'].mount_jacobian(q)
	 jac_df.loc['link','link']=bodies['link'].unity_jacobian(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([3 *[None]],index=['loc_rev', 'ground', 'link'])
def eq(q,bodies,joints,actuators): 
	 eq_s['loc_rev']=joints['loc_rev'].equations(q)
	 eq_s['ground']=bodies['ground'].mount_equation(q)
	 eq_s['link']=bodies['link'].unity_equation(q)
	 system=sparse.bmat(eq_s.values.reshape((3,1)),format='csc') 
	 return system.A.reshape((13,)) 


vel_rhs_s=pd.Series([3 *[None]],index=['loc_rev', 'ground', 'link'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((13,1))
	 return vrhs 
acc_rhs_s=pd.Series([3 *[None]],index=['loc_rev', 'ground', 'link'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['loc_rev']=joints['loc_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['ground']=bodies['ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['link']=bodies['link'].acc_rhs(qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((3,1)),format='csc') 
	 return system.A.reshape((13,)) 


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


JR_s=pd.Series(np.zeros((6)),index=['loc_rev_Fx', 'loc_rev_Fy', 'loc_rev_Fz', 'loc_rev_Mx', 'loc_rev_My', 'loc_rev_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['loc_rev_Fx', 'loc_rev_Fy', 'loc_rev_Fz', 'loc_rev_Mx', 'loc_rev_My', 'loc_rev_Mz']]=joints['loc_rev'].reactions(q,lamda)
	 return JR_s 


