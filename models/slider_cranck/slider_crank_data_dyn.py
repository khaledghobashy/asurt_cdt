import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C_trans', 'jcs_O_rev', 'rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['jcs_A_sph','rbs_cranck']=joints['jcs_A_sph'].jacobian_i(q)
	 jac_df.loc['jcs_A_sph','rbs_connecting']=joints['jcs_A_sph'].jacobian_j(q)
	 jac_df.loc['jcs_B_uni','rbs_connecting']=joints['jcs_B_uni'].jacobian_i(q)
	 jac_df.loc['jcs_B_uni','rbs_slider']=joints['jcs_B_uni'].jacobian_j(q)
	 jac_df.loc['jcs_C_trans','rbs_slider']=joints['jcs_C_trans'].jacobian_i(q)
	 jac_df.loc['jcs_C_trans','rbs_ground']=joints['jcs_C_trans'].jacobian_j(q)
	 jac_df.loc['jcs_O_rev','rbs_cranck']=joints['jcs_O_rev'].jacobian_i(q)
	 jac_df.loc['jcs_O_rev','rbs_ground']=joints['jcs_O_rev'].jacobian_j(q)
	 jac_df.loc['rbs_connecting','rbs_connecting']=bodies['rbs_connecting'].unity_jacobian(q)
	 jac_df.loc['rbs_cranck','rbs_cranck']=bodies['rbs_cranck'].unity_jacobian(q)
	 jac_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mount_jacobian(q)
	 jac_df.loc['rbs_slider','rbs_slider']=bodies['rbs_slider'].unity_jacobian(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([8 *[None]],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C_trans', 'jcs_O_rev', 'rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def eq(q,bodies,joints,actuators): 
	 eq_s['jcs_A_sph']=joints['jcs_A_sph'].equations(q)
	 eq_s['jcs_B_uni']=joints['jcs_B_uni'].equations(q)
	 eq_s['jcs_C_trans']=joints['jcs_C_trans'].equations(q)
	 eq_s['jcs_O_rev']=joints['jcs_O_rev'].equations(q)
	 eq_s['rbs_connecting']=bodies['rbs_connecting'].unity_equation(q)
	 eq_s['rbs_cranck']=bodies['rbs_cranck'].unity_equation(q)
	 eq_s['rbs_ground']=bodies['rbs_ground'].mount_equation(q)
	 eq_s['rbs_slider']=bodies['rbs_slider'].unity_equation(q)
	 system=sparse.bmat(eq_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((27,)) 


vel_rhs_s=pd.Series([8 *[None]],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C_trans', 'jcs_O_rev', 'rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 return vrhs 
acc_rhs_s=pd.Series([8 *[None]],index=['jcs_A_sph', 'jcs_B_uni', 'jcs_C_trans', 'jcs_O_rev', 'rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['jcs_A_sph']=joints['jcs_A_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_B_uni']=joints['jcs_B_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_C_trans']=joints['jcs_C_trans'].acc_rhs(q,qdot)
	 acc_rhs_s['jcs_O_rev']=joints['jcs_O_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['rbs_connecting']=bodies['rbs_connecting'].acc_rhs(qdot)
	 acc_rhs_s['rbs_cranck']=bodies['rbs_cranck'].acc_rhs(qdot)
	 acc_rhs_s['rbs_ground']=bodies['rbs_ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['rbs_slider']=bodies['rbs_slider'].acc_rhs(qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((27,)) 


mass_matrix_df=pd.DataFrame([4 *[None]],columns=['rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'],index=['rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['rbs_connecting','rbs_connecting']=bodies['rbs_connecting'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_cranck','rbs_cranck']=bodies['rbs_cranck'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_ground','rbs_ground']=bodies['rbs_ground'].mass_matrix(q)
	 mass_matrix_df.loc['rbs_slider','rbs_slider']=bodies['rbs_slider'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def Qg(bodies): 
	 Qg_s['rbs_connecting']=bodies['rbs_connecting'].gravity()
	 Qg_s['rbs_cranck']=bodies['rbs_cranck'].gravity()
	 Qg_s['rbs_ground']=bodies['rbs_ground'].gravity()
	 Qg_s['rbs_slider']=bodies['rbs_slider'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qv_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def Qv(bodies,q,qdot): 
	 Qv_s['rbs_connecting']=bodies['rbs_connecting'].centrifugal(q,qdot)
	 Qv_s['rbs_cranck']=bodies['rbs_cranck'].centrifugal(q,qdot)
	 Qv_s['rbs_ground']=bodies['rbs_ground'].centrifugal(q,qdot)
	 Qv_s['rbs_slider']=bodies['rbs_slider'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qa_s=pd.Series([4 *np.zeros((7,1))],index=['rbs_connecting', 'rbs_cranck', 'rbs_ground', 'rbs_slider'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


JR_s=pd.Series(np.zeros((24)),index=['jcs_A_sph_Fx', 'jcs_A_sph_Fy', 'jcs_A_sph_Fz', 'jcs_A_sph_Mx', 'jcs_A_sph_My', 'jcs_A_sph_Mz', 'jcs_B_uni_Fx', 'jcs_B_uni_Fy', 'jcs_B_uni_Fz', 'jcs_B_uni_Mx', 'jcs_B_uni_My', 'jcs_B_uni_Mz', 'jcs_C_trans_Fx', 'jcs_C_trans_Fy', 'jcs_C_trans_Fz', 'jcs_C_trans_Mx', 'jcs_C_trans_My', 'jcs_C_trans_Mz', 'jcs_O_rev_Fx', 'jcs_O_rev_Fy', 'jcs_O_rev_Fz', 'jcs_O_rev_Mx', 'jcs_O_rev_My', 'jcs_O_rev_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['jcs_A_sph_Fx', 'jcs_A_sph_Fy', 'jcs_A_sph_Fz', 'jcs_A_sph_Mx', 'jcs_A_sph_My', 'jcs_A_sph_Mz']]=joints['jcs_A_sph'].reactions(q,lamda)
	 JR_s[['jcs_B_uni_Fx', 'jcs_B_uni_Fy', 'jcs_B_uni_Fz', 'jcs_B_uni_Mx', 'jcs_B_uni_My', 'jcs_B_uni_Mz']]=joints['jcs_B_uni'].reactions(q,lamda)
	 JR_s[['jcs_C_trans_Fx', 'jcs_C_trans_Fy', 'jcs_C_trans_Fz', 'jcs_C_trans_Mx', 'jcs_C_trans_My', 'jcs_C_trans_Mz']]=joints['jcs_C_trans'].reactions(q,lamda)
	 JR_s[['jcs_O_rev_Fx', 'jcs_O_rev_Fy', 'jcs_O_rev_Fz', 'jcs_O_rev_Mx', 'jcs_O_rev_My', 'jcs_O_rev_Mz']]=joints['jcs_O_rev'].reactions(q,lamda)
	 return JR_s 


