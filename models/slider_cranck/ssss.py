import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([2 *[None]],columns=['rbs_ground', 'rbs_link'],index=['jcl_rev', 'jcr_rev', 'jcs_rev', 'rbs_ground', 'rbs_link', 'mcs_rot'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['jcl_rev','rbs_ground']=joints['jcl_rev'].jacobian_i(q)
	 jac_df.loc['jcl_rev','rbs_link']=joints['jcl_rev'].jacobian_j(q)
	 jac_df.loc['jcr_rev','rbs_ground']=joints['jcr_rev'].jacobian_i(q)
	 jac_df.loc['jcr_rev','rbs_link']=joints['jcr_rev'].jacobian_j(q)
	 jac_df.loc['jcs_rev','rbs_ground']=joints['jcs_rev'].jacobian_i(q)
	 jac_df.loc['jcs_rev','rbs_link']=joints['jcs_rev'].jacobian_j(q)
	 jac_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mount_jacobian(q)
	 jac_df.loc['rbs_link','rbs_link']=bodies['rbs_link'].unity_jacobian(q)
	 jac_df.loc['mcs_rot','rbs_ground']=actuators['mcs_rot'].jacobian_i(q)
	 jac_df.loc['mcs_rot','rbs_link']=actuators['mcs_rot'].jacobian_j(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([6 *[None]],index=['jcl_rev', 'jcr_rev', 'jcs_rev', 'rbs_ground', 'rbs_link', 'mcs_rot'])
def eq(q,bodies,joints,actuators): 
	 eq_s['jcl_rev']=joints['jcl_rev'].equations(q)
	 eq_s['jcr_rev']=joints['jcr_rev'].equations(q)
	 eq_s['jcs_rev']=joints['jcs_rev'].equations(q)
	 eq_s['rbs_ground']=bodies['rbs_ground'].mount_equation(q)
	 eq_s['rbs_link']=bodies['rbs_link'].unity_equation(q)
	 eq_s['mcs_rot']=actuators['mcs_rot'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((6,1)),format='csc') 
	 return system.A.reshape((24,)) 


vel_rhs_s=pd.Series([6 *[None]],index=['jcl_rev', 'jcr_rev', 'jcs_rev', 'rbs_ground', 'rbs_link', 'mcs_rot'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((23,1))
	 vrhs=np.concatenate([vrhs,actuators['mcs_rot'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([6 *[None]],index=['jcl_rev', 'jcr_rev', 'jcs_rev', 'rbs_ground', 'rbs_link', 'mcs_rot'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['jcl_rev']=joints['jcl_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['jcr_rev']=joints['jcr_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_rev']=joints['jcs_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['rbs_ground']=bodies['rbs_ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['rbs_link']=bodies['rbs_link'].acc_rhs(qdot)
	 acc_rhs_s['mcs_rot']=actuators['mcs_rot'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((6,1)),format='csc') 
	 return system.A.reshape((24,)) 


mass_matrix_df=pd.DataFrame([2 *[None]],columns=['rbs_ground', 'rbs_link'],index=['rbs_ground', 'rbs_link'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_link','rbs_link']=bodies['rbs_link'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([2 *np.zeros((7,1))],index=['rbs_ground', 'rbs_link'])
def Qg(bodies): 
	 Qg_s['rbs_ground']=bodies['rbs_ground'].gravity()
	 Qg_s['rbs_link']=bodies['rbs_link'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((2,1)),format='csc') 
	 return system.A.reshape((14,)) 


Qv_s=pd.Series([2 *np.zeros((7,1))],index=['rbs_ground', 'rbs_link'])
def Qv(bodies,q,qdot): 
	 Qv_s['rbs_ground']=bodies['rbs_ground'].centrifugal(q,qdot)
	 Qv_s['rbs_link']=bodies['rbs_link'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((2,1)),format='csc') 
	 return system.A.reshape((14,)) 


Qa_s=pd.Series([2 *np.zeros((7,1))],index=['rbs_ground', 'rbs_link'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((2,1)),format='csc') 
	 return system.A.reshape((14,)) 


JR_s=pd.Series(np.zeros((18)),index=['jcl_rev_Fx', 'jcl_rev_Fy', 'jcl_rev_Fz', 'jcl_rev_Mx', 'jcl_rev_My', 'jcl_rev_Mz', 'jcr_rev_Fx', 'jcr_rev_Fy', 'jcr_rev_Fz', 'jcr_rev_Mx', 'jcr_rev_My', 'jcr_rev_Mz', 'jcs_rev_Fx', 'jcs_rev_Fy', 'jcs_rev_Fz', 'jcs_rev_Mx', 'jcs_rev_My', 'jcs_rev_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['jcl_rev_Fx', 'jcl_rev_Fy', 'jcl_rev_Fz', 'jcl_rev_Mx', 'jcl_rev_My', 'jcl_rev_Mz']]=joints['jcl_rev'].reactions(q,lamda)
	 JR_s[['jcr_rev_Fx', 'jcr_rev_Fy', 'jcr_rev_Fz', 'jcr_rev_Mx', 'jcr_rev_My', 'jcr_rev_Mz']]=joints['jcr_rev'].reactions(q,lamda)
	 JR_s[['jcs_rev_Fx', 'jcs_rev_Fy', 'jcs_rev_Fz', 'jcs_rev_Mx', 'jcs_rev_My', 'jcs_rev_Mz']]=joints['jcs_rev'].reactions(q,lamda)
	 return JR_s 


