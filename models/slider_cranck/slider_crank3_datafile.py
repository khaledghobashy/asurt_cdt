import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider'],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C-trans', 'jcs_O_rev', 'rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider', 'mcs_slider_act'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['jcs_A_sph','rbs_crank']=joints['jcs_A_sph'].jacobian_i(q)
	 jac_df.loc['jcs_A_sph','rbs_connecting_rod']=joints['jcs_A_sph'].jacobian_j(q)
	 jac_df.loc['jcs_B_uni','rbs_connecting_rod']=joints['jcs_B_uni'].jacobian_i(q)
	 jac_df.loc['jcs_B_uni','rbs_slider']=joints['jcs_B_uni'].jacobian_j(q)
	 jac_df.loc['jcs_C-trans','rbs_slider']=joints['jcs_C-trans'].jacobian_i(q)
	 jac_df.loc['jcs_C-trans','rbs_ground']=joints['jcs_C-trans'].jacobian_j(q)
	 jac_df.loc['jcs_O_rev','rbs_crank']=joints['jcs_O_rev'].jacobian_i(q)
	 jac_df.loc['jcs_O_rev','rbs_ground']=joints['jcs_O_rev'].jacobian_j(q)
	 jac_df.loc['rbs_connecting_rod','rbs_connecting_rod']=bodies['rbs_connecting_rod'].unity_jacobian(q)
	 jac_df.loc['rbs_crank','rbs_crank']=bodies['rbs_crank'].unity_jacobian(q)
	 jac_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mount_jacobian(q)
	 jac_df.loc['rbs_slider','rbs_slider']=bodies['rbs_slider'].unity_jacobian(q)
	 jac_df.loc['mcs_slider_act','rbs_slider']=actuators['mcs_slider_act'].jacobian_i(q)
	 jac_df.loc['mcs_slider_act','rbs_ground']=actuators['mcs_slider_act'].jacobian_j(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([9 *[None]],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C-trans', 'jcs_O_rev', 'rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider', 'mcs_slider_act'])
def eq(q,bodies,joints,actuators): 
	 eq_s['jcs_A_sph']=joints['jcs_A_sph'].equations(q)
	 eq_s['jcs_B_uni']=joints['jcs_B_uni'].equations(q)
	 eq_s['jcs_C-trans']=joints['jcs_C-trans'].equations(q)
	 eq_s['jcs_O_rev']=joints['jcs_O_rev'].equations(q)
	 eq_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].unity_equation(q)
	 eq_s['rbs_crank']=bodies['rbs_crank'].unity_equation(q)
	 eq_s['rbs_ground']=bodies['rbs_ground'].mount_equation(q)
	 eq_s['rbs_slider']=bodies['rbs_slider'].unity_equation(q)
	 eq_s['mcs_slider_act']=actuators['mcs_slider_act'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


vel_rhs_s=pd.Series([9 *[None]],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C-trans', 'jcs_O_rev', 'rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider', 'mcs_slider_act'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 vrhs=np.concatenate([vrhs,actuators['mcs_slider_act'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([9 *[None]],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C-trans', 'jcs_O_rev', 'rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider', 'mcs_slider_act'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['jcs_A_sph']=joints['jcs_A_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_B_uni']=joints['jcs_B_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_C-trans']=joints['jcs_C-trans'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_O_rev']=joints['jcs_O_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].acc_rhs(qdot)
	 acc_rhs_s['rbs_crank']=bodies['rbs_crank'].acc_rhs(qdot)
	 acc_rhs_s['rbs_ground']=bodies['rbs_ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['rbs_slider']=bodies['rbs_slider'].acc_rhs(qdot)
	 acc_rhs_s['mcs_slider_act']=actuators['mcs_slider_act'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


mass_matrix_df=pd.DataFrame([4 *[None]],columns=['rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider'],index=['rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['rbs_connecting_rod','rbs_connecting_rod']=bodies['rbs_connecting_rod'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_crank','rbs_crank']=bodies['rbs_crank'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_slider','rbs_slider']=bodies['rbs_slider'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider'])
def Qg(bodies): 
	 Qg_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].gravity()
	 Qg_s['rbs_crank']=bodies['rbs_crank'].gravity()
	 Qg_s['rbs_ground']=bodies['rbs_ground'].gravity()
	 Qg_s['rbs_slider']=bodies['rbs_slider'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qv_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider'])
def Qv(bodies,q,qdot): 
	 Qv_s['rbs_connecting_rod']=bodies['rbs_connecting_rod'].centrifugal(q,qdot)
	 Qv_s['rbs_crank']=bodies['rbs_crank'].centrifugal(q,qdot)
	 Qv_s['rbs_ground']=bodies['rbs_ground'].centrifugal(q,qdot)
	 Qv_s['rbs_slider']=bodies['rbs_slider'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qa_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting_rod', 'rbs_crank', 'rbs_ground', 'rbs_slider'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


JR_s=pd.Series(np.zeros((24)),index=['jcs_A_sph_Fx', 'jcs_A_sph_Fy', 'jcs_A_sph_Fz', 'jcs_A_sph_Mx', 'jcs_A_sph_My', 'jcs_A_sph_Mz', 'jcs_B_uni_Fx', 'jcs_B_uni_Fy', 'jcs_B_uni_Fz', 'jcs_B_uni_Mx', 'jcs_B_uni_My', 'jcs_B_uni_Mz', 'jcs_C-trans_Fx', 'jcs_C-trans_Fy', 'jcs_C-trans_Fz', 'jcs_C-trans_Mx', 'jcs_C-trans_My', 'jcs_C-trans_Mz', 'jcs_O_rev_Fx', 'jcs_O_rev_Fy', 'jcs_O_rev_Fz', 'jcs_O_rev_Mx', 'jcs_O_rev_My', 'jcs_O_rev_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['jcs_A_sph_Fx', 'jcs_A_sph_Fy', 'jcs_A_sph_Fz', 'jcs_A_sph_Mx', 'jcs_A_sph_My', 'jcs_A_sph_Mz']]=joints['jcs_A_sph'].reactions(q,lamda)
	 JR_s[['jcs_B_uni_Fx', 'jcs_B_uni_Fy', 'jcs_B_uni_Fz', 'jcs_B_uni_Mx', 'jcs_B_uni_My', 'jcs_B_uni_Mz']]=joints['jcs_B_uni'].reactions(q,lamda)
	 JR_s[['jcs_C-trans_Fx', 'jcs_C-trans_Fy', 'jcs_C-trans_Fz', 'jcs_C-trans_Mx', 'jcs_C-trans_My', 'jcs_C-trans_Mz']]=joints['jcs_C-trans'].reactions(q,lamda)
	 JR_s[['jcs_O_rev_Fx', 'jcs_O_rev_Fy', 'jcs_O_rev_Fz', 'jcs_O_rev_Mx', 'jcs_O_rev_My', 'jcs_O_rev_Mz']]=joints['jcs_O_rev'].reactions(q,lamda)
	 return JR_s 


