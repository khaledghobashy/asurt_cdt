import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler'],index=['jcl_rocker_rev', 'jcr_rocker_rev', 'jcs_rocker_coupler_sph', 'jcs_rocker_coupler_uni', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['jcl_rocker_rev','rbl_rocker']=joints['jcl_rocker_rev'].jacobian_i(q)
	 jac_df.loc['jcl_rocker_rev','rbs_chassis']=joints['jcl_rocker_rev'].jacobian_j(q)
	 jac_df.loc['jcr_rocker_rev','rbr_rocker']=joints['jcr_rocker_rev'].jacobian_i(q)
	 jac_df.loc['jcr_rocker_rev','rbs_chassis']=joints['jcr_rocker_rev'].jacobian_j(q)
	 jac_df.loc['jcs_rocker_coupler_sph','rbl_rocker']=joints['jcs_rocker_coupler_sph'].jacobian_i(q)
	 jac_df.loc['jcs_rocker_coupler_sph','rbs_coupler']=joints['jcs_rocker_coupler_sph'].jacobian_j(q)
	 jac_df.loc['jcs_rocker_coupler_uni','rbr_rocker']=joints['jcs_rocker_coupler_uni'].jacobian_i(q)
	 jac_df.loc['jcs_rocker_coupler_uni','rbs_coupler']=joints['jcs_rocker_coupler_uni'].jacobian_j(q)
	 jac_df.loc['rbl_rocker','rbl_rocker']=bodies['rbl_rocker'].unity_jacobian(q)
	 jac_df.loc['rbr_rocker','rbr_rocker']=bodies['rbr_rocker'].unity_jacobian(q)
	 jac_df.loc['rbs_chassis','rbs_chassis']=bodies['rbs_chassis'].mount_jacobian(q)
	 jac_df.loc['rbs_coupler','rbs_coupler']=bodies['rbs_coupler'].unity_jacobian(q)
	 jac_df.loc['act','rbl_rocker']=actuators['act'].jacobian()
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([9 *[None]],index=['jcl_rocker_rev', 'jcr_rocker_rev', 'jcs_rocker_coupler_sph', 'jcs_rocker_coupler_uni', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def eq(q,bodies,joints,actuators): 
	 eq_s['jcl_rocker_rev']=joints['jcl_rocker_rev'].equations(q)
	 eq_s['jcr_rocker_rev']=joints['jcr_rocker_rev'].equations(q)
	 eq_s['jcs_rocker_coupler_sph']=joints['jcs_rocker_coupler_sph'].equations(q)
	 eq_s['jcs_rocker_coupler_uni']=joints['jcs_rocker_coupler_uni'].equations(q)
	 eq_s['rbl_rocker']=bodies['rbl_rocker'].unity_equation(q)
	 eq_s['rbr_rocker']=bodies['rbr_rocker'].unity_equation(q)
	 eq_s['rbs_chassis']=bodies['rbs_chassis'].mount_equation(q)
	 eq_s['rbs_coupler']=bodies['rbs_coupler'].unity_equation(q)
	 eq_s['act']=actuators['act'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


vel_rhs_s=pd.Series([9 *[None]],index=['jcl_rocker_rev', 'jcr_rocker_rev', 'jcs_rocker_coupler_sph', 'jcs_rocker_coupler_uni', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 vrhs=np.concatenate([vrhs,actuators['act'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([9 *[None]],index=['jcl_rocker_rev', 'jcr_rocker_rev', 'jcs_rocker_coupler_sph', 'jcs_rocker_coupler_uni', 'rbl_rocker', 'rbr_rocker', 'rbs_chassis', 'rbs_coupler', 'act'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['jcl_rocker_rev']=joints['jcl_rocker_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['jcr_rocker_rev']=joints['jcr_rocker_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_rocker_coupler_sph']=joints['jcs_rocker_coupler_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_rocker_coupler_uni']=joints['jcs_rocker_coupler_uni'].acc_rhs(q,qdot)
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


JR_s=pd.Series(np.zeros((24)),index=['jcl_rocker_rev_Fx', 'jcl_rocker_rev_Fy', 'jcl_rocker_rev_Fz', 'jcl_rocker_rev_Mx', 'jcl_rocker_rev_My', 'jcl_rocker_rev_Mz', 'jcr_rocker_rev_Fx', 'jcr_rocker_rev_Fy', 'jcr_rocker_rev_Fz', 'jcr_rocker_rev_Mx', 'jcr_rocker_rev_My', 'jcr_rocker_rev_Mz', 'jcs_rocker_coupler_sph_Fx', 'jcs_rocker_coupler_sph_Fy', 'jcs_rocker_coupler_sph_Fz', 'jcs_rocker_coupler_sph_Mx', 'jcs_rocker_coupler_sph_My', 'jcs_rocker_coupler_sph_Mz', 'jcs_rocker_coupler_uni_Fx', 'jcs_rocker_coupler_uni_Fy', 'jcs_rocker_coupler_uni_Fz', 'jcs_rocker_coupler_uni_Mx', 'jcs_rocker_coupler_uni_My', 'jcs_rocker_coupler_uni_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['jcl_rocker_rev_Fx', 'jcl_rocker_rev_Fy', 'jcl_rocker_rev_Fz', 'jcl_rocker_rev_Mx', 'jcl_rocker_rev_My', 'jcl_rocker_rev_Mz']]=joints['jcl_rocker_rev'].reactions(q,lamda)
	 JR_s[['jcr_rocker_rev_Fx', 'jcr_rocker_rev_Fy', 'jcr_rocker_rev_Fz', 'jcr_rocker_rev_Mx', 'jcr_rocker_rev_My', 'jcr_rocker_rev_Mz']]=joints['jcr_rocker_rev'].reactions(q,lamda)
	 JR_s[['jcs_rocker_coupler_sph_Fx', 'jcs_rocker_coupler_sph_Fy', 'jcs_rocker_coupler_sph_Fz', 'jcs_rocker_coupler_sph_Mx', 'jcs_rocker_coupler_sph_My', 'jcs_rocker_coupler_sph_Mz']]=joints['jcs_rocker_coupler_sph'].reactions(q,lamda)
	 JR_s[['jcs_rocker_coupler_uni_Fx', 'jcs_rocker_coupler_uni_Fy', 'jcs_rocker_coupler_uni_Fz', 'jcs_rocker_coupler_uni_Mx', 'jcs_rocker_coupler_uni_My', 'jcs_rocker_coupler_uni_Mz']]=joints['jcs_rocker_coupler_uni'].reactions(q,lamda)
	 return JR_s 


