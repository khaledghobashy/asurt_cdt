import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'],index=['jcl_rocker_chassis', 'jcr_rocker_chassis', 'jcs_left_rocker_coupler', 'jcs_right_rocker_coupler', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['jcl_rocker_chassis','rbl_rocker']=joints['jcl_rocker_chassis'].jacobian_i(q)
	 jac_df.loc['jcl_rocker_chassis','rbs_chassis']=joints['jcl_rocker_chassis'].jacobian_j(q)
	 jac_df.loc['jcr_rocker_chassis','rbr_rocker']=joints['jcr_rocker_chassis'].jacobian_i(q)
	 jac_df.loc['jcr_rocker_chassis','rbs_chassis']=joints['jcr_rocker_chassis'].jacobian_j(q)
	 jac_df.loc['jcs_left_rocker_coupler','rbl_rocker']=joints['jcs_left_rocker_coupler'].jacobian_i(q)
	 jac_df.loc['jcs_left_rocker_coupler','rbs_coupler']=joints['jcs_left_rocker_coupler'].jacobian_j(q)
	 jac_df.loc['jcs_right_rocker_coupler','rbr_rocker']=joints['jcs_right_rocker_coupler'].jacobian_i(q)
	 jac_df.loc['jcs_right_rocker_coupler','rbs_coupler']=joints['jcs_right_rocker_coupler'].jacobian_j(q)
	 jac_df.loc['rbl_rocker','rbl_rocker']=bodies['rbl_rocker'].unity_jacobian(q)
	 jac_df.loc['rbr_rocker','rbr_rocker']=bodies['rbr_rocker'].unity_jacobian(q)
	 jac_df.loc['rbs_chassis','rbs_chassis']=bodies['rbs_chassis'].mount_jacobian(q)
	 jac_df.loc['rbs_coupler','rbs_coupler']=bodies['rbs_coupler'].unity_jacobian(q)
	 jac_df.loc['act','rbl_rocker']=actuators['act'].jacobian_i(q)
	 jac_df.loc['act','rbs_chassis']=actuators['act'].jacobian_j(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([9 *[None]],index=['jcl_rocker_chassis', 'jcr_rocker_chassis', 'jcs_left_rocker_coupler', 'jcs_right_rocker_coupler', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def eq(q,bodies,joints,actuators): 
	 eq_s['jcl_rocker_chassis']=joints['jcl_rocker_chassis'].equations(q)
	 eq_s['jcr_rocker_chassis']=joints['jcr_rocker_chassis'].equations(q)
	 eq_s['jcs_left_rocker_coupler']=joints['jcs_left_rocker_coupler'].equations(q)
	 eq_s['jcs_right_rocker_coupler']=joints['jcs_right_rocker_coupler'].equations(q)
	 eq_s['rbl_rocker']=bodies['rbl_rocker'].unity_equation(q)
	 eq_s['rbr_rocker']=bodies['rbr_rocker'].unity_equation(q)
	 eq_s['rbs_chassis']=bodies['rbs_chassis'].mount_equation(q)
	 eq_s['rbs_coupler']=bodies['rbs_coupler'].unity_equation(q)
	 eq_s['act']=actuators['act'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


vel_rhs_s=pd.Series([9 *[None]],index=['jcl_rocker_chassis', 'jcr_rocker_chassis', 'jcs_left_rocker_coupler', 'jcs_right_rocker_coupler', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 vrhs=np.concatenate([vrhs,actuators['act'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([9 *[None]],index=['jcl_rocker_chassis', 'jcr_rocker_chassis', 'jcs_left_rocker_coupler', 'jcs_right_rocker_coupler', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['jcl_rocker_chassis']=joints['jcl_rocker_chassis'].acc_rhs(q,qdot)
	 acc_rhs_s['jcr_rocker_chassis']=joints['jcr_rocker_chassis'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_left_rocker_coupler']=joints['jcs_left_rocker_coupler'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_right_rocker_coupler']=joints['jcs_right_rocker_coupler'].acc_rhs(q,qdot)
	 acc_rhs_s['rbl_rocker']=bodies['rbl_rocker'].acc_rhs(qdot)
	 acc_rhs_s['rbr_rocker']=bodies['rbr_rocker'].acc_rhs(qdot)
	 acc_rhs_s['rbs_chassis']=bodies['rbs_chassis'].mount_acc_rhs(qdot)
	 acc_rhs_s['rbs_coupler']=bodies['rbs_coupler'].acc_rhs(qdot)
	 acc_rhs_s['act']=actuators['act'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


mass_matrix_df=pd.DataFrame([4 *[None]],columns=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'],index=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['rbl_rocker','rbl_rocker']=bodies['rbl_rocker'].mass_matrix(q)
	 mass_matrix_df.loc['rbr_rocker','rbr_rocker']=bodies['rbr_rocker'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_chassis','rbs_chassis']=bodies['rbs_chassis'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_coupler','rbs_coupler']=bodies['rbs_coupler'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([4 *np.zeros((7,1))],index=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'])
def Qg(bodies): 
	 Qg_s['rbl_rocker']=bodies['rbl_rocker'].gravity()
	 Qg_s['rbr_rocker']=bodies['rbr_rocker'].gravity()
	 Qg_s['rbs_chassis']=bodies['rbs_chassis'].gravity()
	 Qg_s['rbs_coupler']=bodies['rbs_coupler'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qv_s=pd.Series([4 *np.zeros((7,1))],index=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'])
def Qv(bodies,q,qdot): 
	 Qv_s['rbl_rocker']=bodies['rbl_rocker'].centrifugal(q,qdot)
	 Qv_s['rbr_rocker']=bodies['rbr_rocker'].centrifugal(q,qdot)
	 Qv_s['rbs_chassis']=bodies['rbs_chassis'].centrifugal(q,qdot)
	 Qv_s['rbs_coupler']=bodies['rbs_coupler'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qa_s=pd.Series([4 *np.zeros((7,1))],index=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


JR_s=pd.Series(np.zeros((24)),index=['jcl_rocker_chassis_Fx', 'jcl_rocker_chassis_Fy', 'jcl_rocker_chassis_Fz', 'jcl_rocker_chassis_Mx', 'jcl_rocker_chassis_My', 'jcl_rocker_chassis_Mz', 'jcr_rocker_chassis_Fx', 'jcr_rocker_chassis_Fy', 'jcr_rocker_chassis_Fz', 'jcr_rocker_chassis_Mx', 'jcr_rocker_chassis_My', 'jcr_rocker_chassis_Mz', 'jcs_left_rocker_coupler_Fx', 'jcs_left_rocker_coupler_Fy', 'jcs_left_rocker_coupler_Fz', 'jcs_left_rocker_coupler_Mx', 'jcs_left_rocker_coupler_My', 'jcs_left_rocker_coupler_Mz', 'jcs_right_rocker_coupler_Fx', 'jcs_right_rocker_coupler_Fy', 'jcs_right_rocker_coupler_Fz', 'jcs_right_rocker_coupler_Mx', 'jcs_right_rocker_coupler_My', 'jcs_right_rocker_coupler_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['jcl_rocker_chassis_Fx', 'jcl_rocker_chassis_Fy', 'jcl_rocker_chassis_Fz', 'jcl_rocker_chassis_Mx', 'jcl_rocker_chassis_My', 'jcl_rocker_chassis_Mz']]=joints['jcl_rocker_chassis'].reactions(q,lamda)
	 JR_s[['jcr_rocker_chassis_Fx', 'jcr_rocker_chassis_Fy', 'jcr_rocker_chassis_Fz', 'jcr_rocker_chassis_Mx', 'jcr_rocker_chassis_My', 'jcr_rocker_chassis_Mz']]=joints['jcr_rocker_chassis'].reactions(q,lamda)
	 JR_s[['jcs_left_rocker_coupler_Fx', 'jcs_left_rocker_coupler_Fy', 'jcs_left_rocker_coupler_Fz', 'jcs_left_rocker_coupler_Mx', 'jcs_left_rocker_coupler_My', 'jcs_left_rocker_coupler_Mz']]=joints['jcs_left_rocker_coupler'].reactions(q,lamda)
	 JR_s[['jcs_right_rocker_coupler_Fx', 'jcs_right_rocker_coupler_Fy', 'jcs_right_rocker_coupler_Fz', 'jcs_right_rocker_coupler_Mx', 'jcs_right_rocker_coupler_My', 'jcs_right_rocker_coupler_Mz']]=joints['jcs_right_rocker_coupler'].reactions(q,lamda)
	 return JR_s 


