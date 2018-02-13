import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([6 *[None]],columns=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'],index=['A_rev', 'D_rev', 'B_uni', 'E_uni', 'F_uni', 'C_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.x_actuator'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['A_rev','l1']=joints['A_rev'].jacobian_i(q)
	 jac_df.loc['A_rev','ground']=joints['A_rev'].jacobian_j(q)
	 jac_df.loc['D_rev','l3']=joints['D_rev'].jacobian_i(q)
	 jac_df.loc['D_rev','ground']=joints['D_rev'].jacobian_j(q)
	 jac_df.loc['B_uni','l1']=joints['B_uni'].jacobian_i(q)
	 jac_df.loc['B_uni','l2']=joints['B_uni'].jacobian_j(q)
	 jac_df.loc['E_uni','l4']=joints['E_uni'].jacobian_i(q)
	 jac_df.loc['E_uni','ground']=joints['E_uni'].jacobian_j(q)
	 jac_df.loc['F_uni','l5']=joints['F_uni'].jacobian_i(q)
	 jac_df.loc['F_uni','l1']=joints['F_uni'].jacobian_j(q)
	 jac_df.loc['C_sph','l2']=joints['C_sph'].jacobian_i(q)
	 jac_df.loc['C_sph','l3']=joints['C_sph'].jacobian_j(q)
	 jac_df.loc['EF_cyl','l4']=joints['EF_cyl'].jacobian_i(q)
	 jac_df.loc['EF_cyl','l5']=joints['EF_cyl'].jacobian_j(q)
	 jac_df.loc['ground','ground']=bodies['ground'].mount_jacobian(q)
	 jac_df.loc['l1','l1']=bodies['l1'].unity_jacobian(q)
	 jac_df.loc['l2','l2']=bodies['l2'].unity_jacobian(q)
	 jac_df.loc['l3','l3']=bodies['l3'].unity_jacobian(q)
	 jac_df.loc['l4','l4']=bodies['l4'].unity_jacobian(q)
	 jac_df.loc['l5','l5']=bodies['l5'].unity_jacobian(q)
	 jac_df.loc['l5.x_actuator','l5']=actuators['l5.x_actuator'].jacobian()
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([14 *[None]],index=['A_rev', 'D_rev', 'B_uni', 'E_uni', 'F_uni', 'C_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.x_actuator'])
def eq(q,bodies,joints,actuators): 
	 eq_s['A_rev']=joints['A_rev'].equations(q)
	 eq_s['D_rev']=joints['D_rev'].equations(q)
	 eq_s['B_uni']=joints['B_uni'].equations(q)
	 eq_s['E_uni']=joints['E_uni'].equations(q)
	 eq_s['F_uni']=joints['F_uni'].equations(q)
	 eq_s['C_sph']=joints['C_sph'].equations(q)
	 eq_s['EF_cyl']=joints['EF_cyl'].equations(q)
	 eq_s['ground']=bodies['ground'].mount_equation(q)
	 eq_s['l1']=bodies['l1'].unity_equation(q)
	 eq_s['l2']=bodies['l2'].unity_equation(q)
	 eq_s['l3']=bodies['l3'].unity_equation(q)
	 eq_s['l4']=bodies['l4'].unity_equation(q)
	 eq_s['l5']=bodies['l5'].unity_equation(q)
	 eq_s['l5.x_actuator']=actuators['l5.x_actuator'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((14,1)),format='csc') 
	 return system.A.reshape((42,)) 


vel_rhs_s=pd.Series([14 *[None]],index=['A_rev', 'D_rev', 'B_uni', 'E_uni', 'F_uni', 'C_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.x_actuator'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((41,1))
	 vrhs=np.concatenate([vrhs,actuators['l5.x_actuator'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([14 *[None]],index=['A_rev', 'D_rev', 'B_uni', 'E_uni', 'F_uni', 'C_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.x_actuator'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['A_rev']=joints['A_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['D_rev']=joints['D_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['B_uni']=joints['B_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['E_uni']=joints['E_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['F_uni']=joints['F_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['C_sph']=joints['C_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['EF_cyl']=joints['EF_cyl'].acc_rhs(q,qdot)
	 acc_rhs_s['ground']=bodies['ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['l1']=bodies['l1'].acc_rhs(qdot)
	 acc_rhs_s['l2']=bodies['l2'].acc_rhs(qdot)
	 acc_rhs_s['l3']=bodies['l3'].acc_rhs(qdot)
	 acc_rhs_s['l4']=bodies['l4'].acc_rhs(qdot)
	 acc_rhs_s['l5']=bodies['l5'].acc_rhs(qdot)
	 acc_rhs_s['l5.x_actuator']=actuators['l5.x_actuator'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((14,1)),format='csc') 
	 return system.A.reshape((42,)) 


mass_matrix_df=pd.DataFrame([6 *[None]],columns=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'],index=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['ground','ground']=bodies['ground'].mass_matrix(q)
	 mass_matrix_df.loc['l1','l1']=bodies['l1'].mass_matrix(q)
	 mass_matrix_df.loc['l2','l2']=bodies['l2'].mass_matrix(q)
	 mass_matrix_df.loc['l3','l3']=bodies['l3'].mass_matrix(q)
	 mass_matrix_df.loc['l4','l4']=bodies['l4'].mass_matrix(q)
	 mass_matrix_df.loc['l5','l5']=bodies['l5'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([6 *np.zeros((7,1))],index=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'])
def Qg(bodies): 
	 Qg_s['ground']=bodies['ground'].gravity()
	 Qg_s['l1']=bodies['l1'].gravity()
	 Qg_s['l2']=bodies['l2'].gravity()
	 Qg_s['l3']=bodies['l3'].gravity()
	 Qg_s['l4']=bodies['l4'].gravity()
	 Qg_s['l5']=bodies['l5'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((6,1)),format='csc') 
	 return system.A.reshape((42,)) 


Qv_s=pd.Series([6 *np.zeros((7,1))],index=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'])
def Qv(bodies,q,qdot): 
	 Qv_s['ground']=bodies['ground'].centrifugal(q,qdot)
	 Qv_s['l1']=bodies['l1'].centrifugal(q,qdot)
	 Qv_s['l2']=bodies['l2'].centrifugal(q,qdot)
	 Qv_s['l3']=bodies['l3'].centrifugal(q,qdot)
	 Qv_s['l4']=bodies['l4'].centrifugal(q,qdot)
	 Qv_s['l5']=bodies['l5'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((6,1)),format='csc') 
	 return system.A.reshape((42,)) 


Qa_s=pd.Series([6 *np.zeros((7,1))],index=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((6,1)),format='csc') 
	 return system.A.reshape((42,)) 


JR_s=pd.Series(np.zeros((42)),index=['A_rev_Fx', 'A_rev_Fy', 'A_rev_Fz', 'A_rev_Mx', 'A_rev_My', 'A_rev_Mz', 'D_rev_Fx', 'D_rev_Fy', 'D_rev_Fz', 'D_rev_Mx', 'D_rev_My', 'D_rev_Mz', 'B_uni_Fx', 'B_uni_Fy', 'B_uni_Fz', 'B_uni_Mx', 'B_uni_My', 'B_uni_Mz', 'E_uni_Fx', 'E_uni_Fy', 'E_uni_Fz', 'E_uni_Mx', 'E_uni_My', 'E_uni_Mz', 'F_uni_Fx', 'F_uni_Fy', 'F_uni_Fz', 'F_uni_Mx', 'F_uni_My', 'F_uni_Mz', 'C_sph_Fx', 'C_sph_Fy', 'C_sph_Fz', 'C_sph_Mx', 'C_sph_My', 'C_sph_Mz', 'EF_cyl_Fx', 'EF_cyl_Fy', 'EF_cyl_Fz', 'EF_cyl_Mx', 'EF_cyl_My', 'EF_cyl_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['A_rev_Fx', 'A_rev_Fy', 'A_rev_Fz', 'A_rev_Mx', 'A_rev_My', 'A_rev_Mz']]=joints['A_rev'].reactions(q,lamda)
	 JR_s[['D_rev_Fx', 'D_rev_Fy', 'D_rev_Fz', 'D_rev_Mx', 'D_rev_My', 'D_rev_Mz']]=joints['D_rev'].reactions(q,lamda)
	 JR_s[['B_uni_Fx', 'B_uni_Fy', 'B_uni_Fz', 'B_uni_Mx', 'B_uni_My', 'B_uni_Mz']]=joints['B_uni'].reactions(q,lamda)
	 JR_s[['E_uni_Fx', 'E_uni_Fy', 'E_uni_Fz', 'E_uni_Mx', 'E_uni_My', 'E_uni_Mz']]=joints['E_uni'].reactions(q,lamda)
	 JR_s[['F_uni_Fx', 'F_uni_Fy', 'F_uni_Fz', 'F_uni_Mx', 'F_uni_My', 'F_uni_Mz']]=joints['F_uni'].reactions(q,lamda)
	 JR_s[['C_sph_Fx', 'C_sph_Fy', 'C_sph_Fz', 'C_sph_Mx', 'C_sph_My', 'C_sph_Mz']]=joints['C_sph'].reactions(q,lamda)
	 JR_s[['EF_cyl_Fx', 'EF_cyl_Fy', 'EF_cyl_Fz', 'EF_cyl_Mx', 'EF_cyl_My', 'EF_cyl_Mz']]=joints['EF_cyl'].reactions(q,lamda)
	 return JR_s 


