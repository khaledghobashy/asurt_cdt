import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([6 *[None]],columns=['ground', 'l1', 'l2', 'l3', 'l4', 'l5'],index=['mount_1_rev', 'mount_2_rev', 'C1_uni', 'E_uni', 'F_uni', 'C2_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.y_actuator'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['mount_1_rev','l1']=joints['mount_1_rev'].jacobian_i(q)
	 jac_df.loc['mount_1_rev','ground']=joints['mount_1_rev'].jacobian_j(q)
	 jac_df.loc['mount_2_rev','l3']=joints['mount_2_rev'].jacobian_i(q)
	 jac_df.loc['mount_2_rev','ground']=joints['mount_2_rev'].jacobian_j(q)
	 jac_df.loc['C1_uni','l1']=joints['C1_uni'].jacobian_i(q)
	 jac_df.loc['C1_uni','l2']=joints['C1_uni'].jacobian_j(q)
	 jac_df.loc['E_uni','l4']=joints['E_uni'].jacobian_i(q)
	 jac_df.loc['E_uni','ground']=joints['E_uni'].jacobian_j(q)
	 jac_df.loc['F_uni','l5']=joints['F_uni'].jacobian_i(q)
	 jac_df.loc['F_uni','l3']=joints['F_uni'].jacobian_j(q)
	 jac_df.loc['C2_sph','l2']=joints['C2_sph'].jacobian_i(q)
	 jac_df.loc['C2_sph','l3']=joints['C2_sph'].jacobian_j(q)
	 jac_df.loc['EF_cyl','l4']=joints['EF_cyl'].jacobian_i(q)
	 jac_df.loc['EF_cyl','l5']=joints['EF_cyl'].jacobian_j(q)
	 jac_df.loc['ground','ground']=bodies['ground'].mount_jacobian(q)
	 jac_df.loc['l1','l1']=bodies['l1'].unity_jacobian(q)
	 jac_df.loc['l2','l2']=bodies['l2'].unity_jacobian(q)
	 jac_df.loc['l3','l3']=bodies['l3'].unity_jacobian(q)
	 jac_df.loc['l4','l4']=bodies['l4'].unity_jacobian(q)
	 jac_df.loc['l5','l5']=bodies['l5'].unity_jacobian(q)
	 jac_df.loc['l5.y_actuator','l5']=actuators['l5.y_actuator'].jacobian()
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([14 *[None]],index=['mount_1_rev', 'mount_2_rev', 'C1_uni', 'E_uni', 'F_uni', 'C2_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.y_actuator'])
def eq(q,bodies,joints,actuators): 
	 eq_s['mount_1_rev']=joints['mount_1_rev'].equations(q)
	 eq_s['mount_2_rev']=joints['mount_2_rev'].equations(q)
	 eq_s['C1_uni']=joints['C1_uni'].equations(q)
	 eq_s['E_uni']=joints['E_uni'].equations(q)
	 eq_s['F_uni']=joints['F_uni'].equations(q)
	 eq_s['C2_sph']=joints['C2_sph'].equations(q)
	 eq_s['EF_cyl']=joints['EF_cyl'].equations(q)
	 eq_s['ground']=bodies['ground'].mount_equation(q)
	 eq_s['l1']=bodies['l1'].unity_equation(q)
	 eq_s['l2']=bodies['l2'].unity_equation(q)
	 eq_s['l3']=bodies['l3'].unity_equation(q)
	 eq_s['l4']=bodies['l4'].unity_equation(q)
	 eq_s['l5']=bodies['l5'].unity_equation(q)
	 eq_s['l5.y_actuator']=actuators['l5.y_actuator'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((14,1)),format='csc') 
	 return system.A.reshape((42,)) 


vel_rhs_s=pd.Series([14 *[None]],index=['mount_1_rev', 'mount_2_rev', 'C1_uni', 'E_uni', 'F_uni', 'C2_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.y_actuator'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((41,1))
	 vrhs=np.concatenate([vrhs,actuators['l5.y_actuator'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([14 *[None]],index=['mount_1_rev', 'mount_2_rev', 'C1_uni', 'E_uni', 'F_uni', 'C2_sph', 'EF_cyl', 'ground', 'l1', 'l2', 'l3', 'l4', 'l5', 'l5.y_actuator'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['mount_1_rev']=joints['mount_1_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['mount_2_rev']=joints['mount_2_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['C1_uni']=joints['C1_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['E_uni']=joints['E_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['F_uni']=joints['F_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['C2_sph']=joints['C2_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['EF_cyl']=joints['EF_cyl'].acc_rhs(q,qdot)
	 acc_rhs_s['ground']=bodies['ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['l1']=bodies['l1'].acc_rhs(qdot)
	 acc_rhs_s['l2']=bodies['l2'].acc_rhs(qdot)
	 acc_rhs_s['l3']=bodies['l3'].acc_rhs(qdot)
	 acc_rhs_s['l4']=bodies['l4'].acc_rhs(qdot)
	 acc_rhs_s['l5']=bodies['l5'].acc_rhs(qdot)
	 acc_rhs_s['l5.y_actuator']=actuators['l5.y_actuator'].acc_rhs(q,qdot)
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


JR_s=pd.Series(np.zeros((42)),index=['mount_1_rev_Fx', 'mount_1_rev_Fy', 'mount_1_rev_Fz', 'mount_1_rev_Mx', 'mount_1_rev_My', 'mount_1_rev_Mz', 'mount_2_rev_Fx', 'mount_2_rev_Fy', 'mount_2_rev_Fz', 'mount_2_rev_Mx', 'mount_2_rev_My', 'mount_2_rev_Mz', 'C1_uni_Fx', 'C1_uni_Fy', 'C1_uni_Fz', 'C1_uni_Mx', 'C1_uni_My', 'C1_uni_Mz', 'E_uni_Fx', 'E_uni_Fy', 'E_uni_Fz', 'E_uni_Mx', 'E_uni_My', 'E_uni_Mz', 'F_uni_Fx', 'F_uni_Fy', 'F_uni_Fz', 'F_uni_Mx', 'F_uni_My', 'F_uni_Mz', 'C2_sph_Fx', 'C2_sph_Fy', 'C2_sph_Fz', 'C2_sph_Mx', 'C2_sph_My', 'C2_sph_Mz', 'EF_cyl_Fx', 'EF_cyl_Fy', 'EF_cyl_Fz', 'EF_cyl_Mx', 'EF_cyl_My', 'EF_cyl_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['mount_1_rev_Fx', 'mount_1_rev_Fy', 'mount_1_rev_Fz', 'mount_1_rev_Mx', 'mount_1_rev_My', 'mount_1_rev_Mz']]=joints['mount_1_rev'].reactions(q,lamda)
	 JR_s[['mount_2_rev_Fx', 'mount_2_rev_Fy', 'mount_2_rev_Fz', 'mount_2_rev_Mx', 'mount_2_rev_My', 'mount_2_rev_Mz']]=joints['mount_2_rev'].reactions(q,lamda)
	 JR_s[['C1_uni_Fx', 'C1_uni_Fy', 'C1_uni_Fz', 'C1_uni_Mx', 'C1_uni_My', 'C1_uni_Mz']]=joints['C1_uni'].reactions(q,lamda)
	 JR_s[['E_uni_Fx', 'E_uni_Fy', 'E_uni_Fz', 'E_uni_Mx', 'E_uni_My', 'E_uni_Mz']]=joints['E_uni'].reactions(q,lamda)
	 JR_s[['F_uni_Fx', 'F_uni_Fy', 'F_uni_Fz', 'F_uni_Mx', 'F_uni_My', 'F_uni_Mz']]=joints['F_uni'].reactions(q,lamda)
	 JR_s[['C2_sph_Fx', 'C2_sph_Fy', 'C2_sph_Fz', 'C2_sph_Mx', 'C2_sph_My', 'C2_sph_Mz']]=joints['C2_sph'].reactions(q,lamda)
	 JR_s[['EF_cyl_Fx', 'EF_cyl_Fy', 'EF_cyl_Fz', 'EF_cyl_Mx', 'EF_cyl_My', 'EF_cyl_Mz']]=joints['EF_cyl'].reactions(q,lamda)
	 return JR_s 


