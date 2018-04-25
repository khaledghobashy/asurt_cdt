import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider'],index=['jcs_cranck_connecting_sph', 'jcs_ground_cranck_rev', 'jcs_slider_connecting_sph', 'jcs_slider_ground_trans', 'rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider', 'act'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['jcs_cranck_connecting_sph','rbs_cranck']=joints['jcs_cranck_connecting_sph'].jacobian_i(q)
	 jac_df.loc['jcs_cranck_connecting_sph','rbs_connecting_rod']=joints['jcs_cranck_connecting_sph'].jacobian_j(q)
	 jac_df.loc['jcs_ground_cranck_rev','rbs_cranck']=joints['jcs_ground_cranck_rev'].jacobian_i(q)
	 jac_df.loc['jcs_ground_cranck_rev','rbs_ground']=joints['jcs_ground_cranck_rev'].jacobian_j(q)
	 jac_df.loc['jcs_slider_connecting_sph','rbs_connecting_rod']=joints['jcs_slider_connecting_sph'].jacobian_i(q)
	 jac_df.loc['jcs_slider_connecting_sph','rbs_slider']=joints['jcs_slider_connecting_sph'].jacobian_j(q)
	 jac_df.loc['jcs_slider_ground_trans','rbs_slider']=joints['jcs_slider_ground_trans'].jacobian_i(q)
	 jac_df.loc['jcs_slider_ground_trans','rbs_ground']=joints['jcs_slider_ground_trans'].jacobian_j(q)
	 jac_df.loc['rbs_connecting_rod','rbs_connecting_rod']=bodies['rbs_connecting_rod'].unity_jacobian(q)
	 jac_df.loc['rbs_cranck','rbs_cranck']=bodies['rbs_cranck'].unity_jacobian(q)
	 jac_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mount_jacobian(q)
	 jac_df.loc['rbs_slider','rbs_slider']=bodies['rbs_slider'].unity_jacobian(q)
	 jac_df.loc['act','rbs_slider']=actuators['act'].jacobian_i(q)
	 jac_df.loc['act','rbs_ground']=actuators['act'].jacobian_j(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([9 *[None]],index=['jcs_cranck_connecting_sph', 'jcs_ground_cranck_rev', 'jcs_slider_connecting_sph', 'jcs_slider_ground_trans', 'rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider', 'act'])
def eq(q,bodies,joints,actuators): 
	 eq_s['jcs_cranck_connecting_sph']=joints['jcs_cranck_connecting_sph'].equations(q)
	 eq_s['jcs_ground_cranck_rev']=joints['jcs_ground_cranck_rev'].equations(q)
	 eq_s['jcs_slider_connecting_sph']=joints['jcs_slider_connecting_sph'].equations(q)
	 eq_s['jcs_slider_ground_trans']=joints['jcs_slider_ground_trans'].equations(q)
	 eq_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].unity_equation(q)
	 eq_s['rbs_cranck']=bodies['rbs_cranck'].unity_equation(q)
	 eq_s['rbs_ground']=bodies['rbs_ground'].mount_equation(q)
	 eq_s['rbs_slider']=bodies['rbs_slider'].unity_equation(q)
	 eq_s['act']=actuators['act'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


vel_rhs_s=pd.Series([9 *[None]],index=['jcs_cranck_connecting_sph', 'jcs_ground_cranck_rev', 'jcs_slider_connecting_sph', 'jcs_slider_ground_trans', 'rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider', 'act'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 vrhs=np.concatenate([vrhs,actuators['act'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([9 *[None]],index=['jcs_cranck_connecting_sph', 'jcs_ground_cranck_rev', 'jcs_slider_connecting_sph', 'jcs_slider_ground_trans', 'rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider', 'act'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['jcs_cranck_connecting_sph']=joints['jcs_cranck_connecting_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_ground_cranck_rev']=joints['jcs_ground_cranck_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_slider_connecting_sph']=joints['jcs_slider_connecting_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_slider_ground_trans']=joints['jcs_slider_ground_trans'].acc_rhs(q,qdot)
	 acc_rhs_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].acc_rhs(qdot)
	 acc_rhs_s['rbs_cranck']=bodies['rbs_cranck'].acc_rhs(qdot)
	 acc_rhs_s['rbs_ground']=bodies['rbs_ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['rbs_slider']=bodies['rbs_slider'].acc_rhs(qdot)
	 acc_rhs_s['act']=actuators['act'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


mass_matrix_df=pd.DataFrame([4 *[None]],columns=['rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider'],index=['rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['rbs_connecting_rod','rbs_connecting_rod']=bodies['rbs_connecting_rod'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_cranck','rbs_cranck']=bodies['rbs_cranck'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_slider','rbs_slider']=bodies['rbs_slider'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def Qg(bodies): 
	 Qg_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].gravity()
	 Qg_s['rbs_cranck']=bodies['rbs_cranck'].gravity()
	 Qg_s['rbs_ground']=bodies['rbs_ground'].gravity()
	 Qg_s['rbs_slider']=bodies['rbs_slider'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qv_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def Qv(bodies,q,qdot): 
	 Qv_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].centrifugal(q,qdot)
	 Qv_s['rbs_cranck']=bodies['rbs_cranck'].centrifugal(q,qdot)
	 Qv_s['rbs_ground']=bodies['rbs_ground'].centrifugal(q,qdot)
	 Qv_s['rbs_slider']=bodies['rbs_slider'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qa_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting_rod', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


JR_s=pd.Series(np.zeros((24)),index=['jcs_cranck_connecting_sph_Fx', 'jcs_cranck_connecting_sph_Fy', 'jcs_cranck_connecting_sph_Fz', 'jcs_cranck_connecting_sph_Mx', 'jcs_cranck_connecting_sph_My', 'jcs_cranck_connecting_sph_Mz', 'jcs_ground_cranck_rev_Fx', 'jcs_ground_cranck_rev_Fy', 'jcs_ground_cranck_rev_Fz', 'jcs_ground_cranck_rev_Mx', 'jcs_ground_cranck_rev_My', 'jcs_ground_cranck_rev_Mz', 'jcs_slider_connecting_sph_Fx', 'jcs_slider_connecting_sph_Fy', 'jcs_slider_connecting_sph_Fz', 'jcs_slider_connecting_sph_Mx', 'jcs_slider_connecting_sph_My', 'jcs_slider_connecting_sph_Mz', 'jcs_slider_ground_trans_Fx', 'jcs_slider_ground_trans_Fy', 'jcs_slider_ground_trans_Fz', 'jcs_slider_ground_trans_Mx', 'jcs_slider_ground_trans_My', 'jcs_slider_ground_trans_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['jcs_cranck_connecting_sph_Fx', 'jcs_cranck_connecting_sph_Fy', 'jcs_cranck_connecting_sph_Fz', 'jcs_cranck_connecting_sph_Mx', 'jcs_cranck_connecting_sph_My', 'jcs_cranck_connecting_sph_Mz']]=joints['jcs_cranck_connecting_sph'].reactions(q,lamda)
	 JR_s[['jcs_ground_cranck_rev_Fx', 'jcs_ground_cranck_rev_Fy', 'jcs_ground_cranck_rev_Fz', 'jcs_ground_cranck_rev_Mx', 'jcs_ground_cranck_rev_My', 'jcs_ground_cranck_rev_Mz']]=joints['jcs_ground_cranck_rev'].reactions(q,lamda)
	 JR_s[['jcs_slider_connecting_sph_Fx', 'jcs_slider_connecting_sph_Fy', 'jcs_slider_connecting_sph_Fz', 'jcs_slider_connecting_sph_Mx', 'jcs_slider_connecting_sph_My', 'jcs_slider_connecting_sph_Mz']]=joints['jcs_slider_connecting_sph'].reactions(q,lamda)
	 JR_s[['jcs_slider_ground_trans_Fx', 'jcs_slider_ground_trans_Fy', 'jcs_slider_ground_trans_Fz', 'jcs_slider_ground_trans_Mx', 'jcs_slider_ground_trans_My', 'jcs_slider_ground_trans_Mz']]=joints['jcs_slider_ground_trans'].reactions(q,lamda)
	 return JR_s 


