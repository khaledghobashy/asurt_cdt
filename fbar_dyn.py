import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['ground', 'l1', 'l2', 'l3'],index=['A_rev', 'D_rev', 'B_uni', 'C_sph', 'ground', 'l1', 'l2', 'l3'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['A_rev','ground']=joints['A_rev'].jacobian_i(q)
	 jac_df.loc['A_rev','l1']=joints['A_rev'].jacobian_j(q)
	 jac_df.loc['D_rev','l3']=joints['D_rev'].jacobian_i(q)
	 jac_df.loc['D_rev','ground']=joints['D_rev'].jacobian_j(q)
	 jac_df.loc['B_uni','l1']=joints['B_uni'].jacobian_i(q)
	 jac_df.loc['B_uni','l2']=joints['B_uni'].jacobian_j(q)
	 jac_df.loc['C_sph','l2']=joints['C_sph'].jacobian_i(q)
	 jac_df.loc['C_sph','l3']=joints['C_sph'].jacobian_j(q)
	 jac_df.loc['ground','ground']=bodies['ground'].mount_jacobian(q)
	 jac_df.loc['l1','l1']=bodies['l1'].unity_jacobian(q)
	 jac_df.loc['l2','l2']=bodies['l2'].unity_jacobian(q)
	 jac_df.loc['l3','l3']=bodies['l3'].unity_jacobian(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([8 *[None]],index=['A_rev', 'D_rev', 'B_uni', 'C_sph', 'ground', 'l1', 'l2', 'l3'])
def eq(q,bodies,joints,actuators): 
	 eq_s['A_rev']=joints['A_rev'].equations(q)
	 eq_s['D_rev']=joints['D_rev'].equations(q)
	 eq_s['B_uni']=joints['B_uni'].equations(q)
	 eq_s['C_sph']=joints['C_sph'].equations(q)
	 eq_s['ground']=bodies['ground'].mount_equation(q)
	 eq_s['l1']=bodies['l1'].unity_equation(q)
	 eq_s['l2']=bodies['l2'].unity_equation(q)
	 eq_s['l3']=bodies['l3'].unity_equation(q)
	 system=sparse.bmat(eq_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((27,)) 


vel_rhs_s=pd.Series([8 *[None]],index=['A_rev', 'D_rev', 'B_uni', 'C_sph', 'ground', 'l1', 'l2', 'l3'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 return vrhs 
acc_rhs_s=pd.Series([8 *[None]],index=['A_rev', 'D_rev', 'B_uni', 'C_sph', 'ground', 'l1', 'l2', 'l3'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['A_rev']=joints['A_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['D_rev']=joints['D_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['B_uni']=joints['B_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['C_sph']=joints['C_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['ground']=bodies['ground'].mount_acc_rhs(qdot)
	 acc_rhs_s['l1']=bodies['l1'].acc_rhs(qdot)
	 acc_rhs_s['l2']=bodies['l2'].acc_rhs(qdot)
	 acc_rhs_s['l3']=bodies['l3'].acc_rhs(qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((27,)) 


mass_matrix_df=pd.DataFrame([4 *[None]],columns=['ground', 'l1', 'l2', 'l3'],index=['ground', 'l1', 'l2', 'l3'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['ground','ground']=bodies['ground'].mass_matrix(q)
	 mass_matrix_df.loc['l1','l1']=bodies['l1'].mass_matrix(q)
	 mass_matrix_df.loc['l2','l2']=bodies['l2'].mass_matrix(q)
	 mass_matrix_df.loc['l3','l3']=bodies['l3'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([4 *np.zeros((7,1))],index=['ground', 'l1', 'l2', 'l3'])
def Qg(bodies): 
	 Qg_s['ground']=bodies['ground'].gravity()
	 Qg_s['l1']=bodies['l1'].gravity()
	 Qg_s['l2']=bodies['l2'].gravity()
	 Qg_s['l3']=bodies['l3'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qv_s=pd.Series([4 *np.zeros((7,1))],index=['ground', 'l1', 'l2', 'l3'])
def Qv(bodies,q,qdot): 
	 Qv_s['ground']=bodies['ground'].centrifugal(q,qdot)
	 Qv_s['l1']=bodies['l1'].centrifugal(q,qdot)
	 Qv_s['l2']=bodies['l2'].centrifugal(q,qdot)
	 Qv_s['l3']=bodies['l3'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


Qa_s=pd.Series([4 *np.zeros((7,1))],index=['ground', 'l1', 'l2', 'l3'])
def Qa(forces,q,qdot): 
	 system=sparse.bmat(Qa_s.values.reshape((4,1)),format='csc') 
	 return system.A.reshape((28,)) 


JR_s=pd.Series([np.zeros((24,1))],index=['A_rev_Fx', 'A_rev_Fy', 'A_rev_Fz', 'A_rev_Mx', 'A_rev_My', 'A_rev_Mz', 'D_rev_Fx', 'D_rev_Fy', 'D_rev_Fz', 'D_rev_Mx', 'D_rev_My', 'D_rev_Mz', 'B_uni_Fx', 'B_uni_Fy', 'B_uni_Fz', 'B_uni_Mx', 'B_uni_My', 'B_uni_Mz', 'C_sph_Fx', 'C_sph_Fy', 'C_sph_Fz', 'C_sph_Mx', 'C_sph_My', 'C_sph_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['A_rev_Fx', 'A_rev_Fy', 'A_rev_Fz', 'A_rev_Mx', 'A_rev_My', 'A_rev_Mz']]=joints['A_rev'].reactions(q,lamda)
	 JR_s[['D_rev_Fx', 'D_rev_Fy', 'D_rev_Fz', 'D_rev_Mx', 'D_rev_My', 'D_rev_Mz']]=joints['D_rev'].reactions(q,lamda)
	 JR_s[['B_uni_Fx', 'B_uni_Fy', 'B_uni_Fz', 'B_uni_Mx', 'B_uni_My', 'B_uni_Mz']]=joints['B_uni'].reactions(q,lamda)
	 JR_s[['C_sph_Fx', 'C_sph_Fy', 'C_sph_Fz', 'C_sph_Mx', 'C_sph_My', 'C_sph_Mz']]=joints['C_sph'].reactions(q,lamda)
	 return JR_s 


