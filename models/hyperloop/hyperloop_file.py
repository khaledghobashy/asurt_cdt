import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([5 *[None]],columns=['chassis', 'wb1', 'wb2', 'wb3', 'wb4'],index=['w1_rev', 'w2_rev', 'w3_rev', 'w4_rev', 'chassis', 'wb1', 'wb2', 'wb3', 'wb4', 'w1_rev_actuator', 'w2_rev_actuator', 'w3_rev_actuator', 'w4_rev_actuator'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['w1_rev','wb1']=joints['w1_rev'].jacobian_i(q)
	 jac_df.loc['w1_rev','chassis']=joints['w1_rev'].jacobian_j(q)
	 jac_df.loc['w2_rev','wb2']=joints['w2_rev'].jacobian_i(q)
	 jac_df.loc['w2_rev','chassis']=joints['w2_rev'].jacobian_j(q)
	 jac_df.loc['w3_rev','wb3']=joints['w3_rev'].jacobian_i(q)
	 jac_df.loc['w3_rev','chassis']=joints['w3_rev'].jacobian_j(q)
	 jac_df.loc['w4_rev','wb4']=joints['w4_rev'].jacobian_i(q)
	 jac_df.loc['w4_rev','chassis']=joints['w4_rev'].jacobian_j(q)
	 jac_df.loc['chassis','chassis']=bodies['chassis'].unity_jacobian(q)
	 jac_df.loc['wb1','wb1']=bodies['wb1'].unity_jacobian(q)
	 jac_df.loc['wb2','wb2']=bodies['wb2'].unity_jacobian(q)
	 jac_df.loc['wb3','wb3']=bodies['wb3'].unity_jacobian(q)
	 jac_df.loc['wb4','wb4']=bodies['wb4'].unity_jacobian(q)
	 jac_df.loc['w1_rev_actuator','wb1']=actuators['w1_rev_actuator'].jacobian_i(q)
	 jac_df.loc['w1_rev_actuator','chassis']=actuators['w1_rev_actuator'].jacobian_j(q)
	 jac_df.loc['w2_rev_actuator','wb2']=actuators['w2_rev_actuator'].jacobian_i(q)
	 jac_df.loc['w2_rev_actuator','chassis']=actuators['w2_rev_actuator'].jacobian_j(q)
	 jac_df.loc['w3_rev_actuator','wb3']=actuators['w3_rev_actuator'].jacobian_i(q)
	 jac_df.loc['w3_rev_actuator','chassis']=actuators['w3_rev_actuator'].jacobian_j(q)
	 jac_df.loc['w4_rev_actuator','wb4']=actuators['w4_rev_actuator'].jacobian_i(q)
	 jac_df.loc['w4_rev_actuator','chassis']=actuators['w4_rev_actuator'].jacobian_j(q)
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([13 *[None]],index=['w1_rev', 'w2_rev', 'w3_rev', 'w4_rev', 'chassis', 'wb1', 'wb2', 'wb3', 'wb4', 'w1_rev_actuator', 'w2_rev_actuator', 'w3_rev_actuator', 'w4_rev_actuator'])
def eq(q,bodies,joints,actuators): 
	 eq_s['w1_rev']=joints['w1_rev'].equations(q)
	 eq_s['w2_rev']=joints['w2_rev'].equations(q)
	 eq_s['w3_rev']=joints['w3_rev'].equations(q)
	 eq_s['w4_rev']=joints['w4_rev'].equations(q)
	 eq_s['chassis']=bodies['chassis'].unity_equation(q)
	 eq_s['wb1']=bodies['wb1'].unity_equation(q)
	 eq_s['wb2']=bodies['wb2'].unity_equation(q)
	 eq_s['wb3']=bodies['wb3'].unity_equation(q)
	 eq_s['wb4']=bodies['wb4'].unity_equation(q)
	 eq_s['w1_rev_actuator']=actuators['w1_rev_actuator'].equations(q)
	 eq_s['w2_rev_actuator']=actuators['w2_rev_actuator'].equations(q)
	 eq_s['w3_rev_actuator']=actuators['w3_rev_actuator'].equations(q)
	 eq_s['w4_rev_actuator']=actuators['w4_rev_actuator'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((13,1)),format='csc') 
	 return system.A.reshape((29,)) 


vel_rhs_s=pd.Series([13 *[None]],index=['w1_rev', 'w2_rev', 'w3_rev', 'w4_rev', 'chassis', 'wb1', 'wb2', 'wb3', 'wb4', 'w1_rev_actuator', 'w2_rev_actuator', 'w3_rev_actuator', 'w4_rev_actuator'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((25,1))
	 vrhs=np.concatenate([vrhs,actuators['w1_rev_actuator'].vel_rhs()]) 
	 vrhs=np.concatenate([vrhs,actuators['w2_rev_actuator'].vel_rhs()]) 
	 vrhs=np.concatenate([vrhs,actuators['w3_rev_actuator'].vel_rhs()]) 
	 vrhs=np.concatenate([vrhs,actuators['w4_rev_actuator'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([13 *[None]],index=['w1_rev', 'w2_rev', 'w3_rev', 'w4_rev', 'chassis', 'wb1', 'wb2', 'wb3', 'wb4', 'w1_rev_actuator', 'w2_rev_actuator', 'w3_rev_actuator', 'w4_rev_actuator'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['w1_rev']=joints['w1_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['w2_rev']=joints['w2_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['w3_rev']=joints['w3_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['w4_rev']=joints['w4_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['chassis']=bodies['chassis'].acc_rhs(qdot)
	 acc_rhs_s['wb1']=bodies['wb1'].acc_rhs(qdot)
	 acc_rhs_s['wb2']=bodies['wb2'].acc_rhs(qdot)
	 acc_rhs_s['wb3']=bodies['wb3'].acc_rhs(qdot)
	 acc_rhs_s['wb4']=bodies['wb4'].acc_rhs(qdot)
	 acc_rhs_s['w1_rev_actuator']=actuators['w1_rev_actuator'].acc_rhs(q,qdot)
	 acc_rhs_s['w2_rev_actuator']=actuators['w2_rev_actuator'].acc_rhs(q,qdot)
	 acc_rhs_s['w3_rev_actuator']=actuators['w3_rev_actuator'].acc_rhs(q,qdot)
	 acc_rhs_s['w4_rev_actuator']=actuators['w4_rev_actuator'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((13,1)),format='csc') 
	 return system.A.reshape((29,)) 


mass_matrix_df=pd.DataFrame([5 *[None]],columns=['chassis', 'wb1', 'wb2', 'wb3', 'wb4'],index=['chassis', 'wb1', 'wb2', 'wb3', 'wb4'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['chassis','chassis']=bodies['chassis'].mass_matrix(q)
	 mass_matrix_df.loc['wb1','wb1']=bodies['wb1'].mass_matrix(q)
	 mass_matrix_df.loc['wb2','wb2']=bodies['wb2'].mass_matrix(q)
	 mass_matrix_df.loc['wb3','wb3']=bodies['wb3'].mass_matrix(q)
	 mass_matrix_df.loc['wb4','wb4']=bodies['wb4'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([5 *np.zeros((7,1))],index=['chassis', 'wb1', 'wb2', 'wb3', 'wb4'])
def Qg(bodies): 
	 Qg_s['chassis']=bodies['chassis'].gravity()
	 Qg_s['wb1']=bodies['wb1'].gravity()
	 Qg_s['wb2']=bodies['wb2'].gravity()
	 Qg_s['wb3']=bodies['wb3'].gravity()
	 Qg_s['wb4']=bodies['wb4'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((5,1)),format='csc') 
	 return system.A.reshape((35,)) 


Qv_s=pd.Series([5 *np.zeros((7,1))],index=['chassis', 'wb1', 'wb2', 'wb3', 'wb4'])
def Qv(bodies,q,qdot): 
	 Qv_s['chassis']=bodies['chassis'].centrifugal(q,qdot)
	 Qv_s['wb1']=bodies['wb1'].centrifugal(q,qdot)
	 Qv_s['wb2']=bodies['wb2'].centrifugal(q,qdot)
	 Qv_s['wb3']=bodies['wb3'].centrifugal(q,qdot)
	 Qv_s['wb4']=bodies['wb4'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((5,1)),format='csc') 
	 return system.A.reshape((35,)) 


Qa_s=pd.Series([5 *np.zeros((7,1))],index=['chassis', 'wb1', 'wb2', 'wb3', 'wb4'])
def Qa(forces,q,qdot): 
	 Qa_s['wb1']=forces['tvf1'].equation(q,qdot)
	 Qa_s['wb2']=forces['tvf2'].equation(q,qdot)
	 Qa_s['wb3']=forces['tvf3'].equation(q,qdot)
	 Qa_s['wb4']=forces['tvf4'].equation(q,qdot)
	 Qa_s['wb1']=forces['brake1'].equation(q,qdot)
	 Qa_s['wb2']=forces['brake2'].equation(q,qdot)
	 Qa_s['wb3']=forces['brake3'].equation(q,qdot)
	 Qa_s['wb4']=forces['brake4'].equation(q,qdot)
	 system=sparse.bmat(Qa_s.values.reshape((5,1)),format='csc') 
	 return system.A.reshape((35,)) 


JR_s=pd.Series(np.zeros((24)),index=['w1_rev_Fx', 'w1_rev_Fy', 'w1_rev_Fz', 'w1_rev_Mx', 'w1_rev_My', 'w1_rev_Mz', 'w2_rev_Fx', 'w2_rev_Fy', 'w2_rev_Fz', 'w2_rev_Mx', 'w2_rev_My', 'w2_rev_Mz', 'w3_rev_Fx', 'w3_rev_Fy', 'w3_rev_Fz', 'w3_rev_Mx', 'w3_rev_My', 'w3_rev_Mz', 'w4_rev_Fx', 'w4_rev_Fy', 'w4_rev_Fz', 'w4_rev_Mx', 'w4_rev_My', 'w4_rev_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['w1_rev_Fx', 'w1_rev_Fy', 'w1_rev_Fz', 'w1_rev_Mx', 'w1_rev_My', 'w1_rev_Mz']]=joints['w1_rev'].reactions(q,lamda)
	 JR_s[['w2_rev_Fx', 'w2_rev_Fy', 'w2_rev_Fz', 'w2_rev_Mx', 'w2_rev_My', 'w2_rev_Mz']]=joints['w2_rev'].reactions(q,lamda)
	 JR_s[['w3_rev_Fx', 'w3_rev_Fy', 'w3_rev_Fz', 'w3_rev_Mx', 'w3_rev_My', 'w3_rev_Mz']]=joints['w3_rev'].reactions(q,lamda)
	 JR_s[['w4_rev_Fx', 'w4_rev_Fy', 'w4_rev_Fz', 'w4_rev_Mx', 'w4_rev_My', 'w4_rev_Mz']]=joints['w4_rev'].reactions(q,lamda)
	 return JR_s 


