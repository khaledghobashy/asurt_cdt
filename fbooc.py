import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([4 *[None]],columns=['ground', 'l1', 'l2', 'l3'],index=['A_rev', 'D_rev', 'B_sph', 'C_uni', 'ground', 'l1', 'l2', 'l3', 'l1.y_actuator'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['A_rev','ground']=joints['A_rev'].jacobian_i(q)
	 jac_df.loc['A_rev','l1']=joints['A_rev'].jacobian_j(q)
	 jac_df.loc['D_rev','l3']=joints['D_rev'].jacobian_i(q)
	 jac_df.loc['D_rev','ground']=joints['D_rev'].jacobian_j(q)
	 jac_df.loc['B_sph','l1']=joints['B_sph'].jacobian_i(q)
	 jac_df.loc['B_sph','l2']=joints['B_sph'].jacobian_j(q)
	 jac_df.loc['C_uni','l2']=joints['C_uni'].jacobian_i(q)
	 jac_df.loc['C_uni','l3']=joints['C_uni'].jacobian_j(q)
	 jac_df.loc['ground','ground']=bodies['ground'].mount_jacobian(q)
	 jac_df.loc['l1','l1']=bodies['l1'].unity_jacobian(q)
	 jac_df.loc['l2','l2']=bodies['l2'].unity_jacobian(q)
	 jac_df.loc['l3','l3']=bodies['l3'].unity_jacobian(q)
	 jac_df.loc['l1.y_actuator','l1']=actuators['l1.y_actuator'].jacobian()
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([9 *[None]],index=['A_rev', 'D_rev', 'B_sph', 'C_uni', 'ground', 'l1', 'l2', 'l3', 'l1.y_actuator'])
def eq(q,bodies,joints,actuators): 
	 eq_s['A_rev']=joints['A_rev'].equations(q)
	 eq_s['D_rev']=joints['D_rev'].equations(q)
	 eq_s['B_sph']=joints['B_sph'].equations(q)
	 eq_s['C_uni']=joints['C_uni'].equations(q)
	 eq_s['ground']=bodies['ground'].mount_equation(q)
	 eq_s['l1']=bodies['l1'].unity_equation(q)
	 eq_s['l2']=bodies['l2'].unity_equation(q)
	 eq_s['l3']=bodies['l3'].unity_equation(q)
	 eq_s['l1.y_actuator']=actuators['l1.y_actuator'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


vel_rhs_s=pd.Series([9 *[None]],index=['A_rev', 'D_rev', 'B_sph', 'C_uni', 'ground', 'l1', 'l2', 'l3', 'l1.y_actuator'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((27,1))
	 vrhs=np.concatenate([vrhs,actuators['l1.y_actuator'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([9 *[None]],index=['A_rev', 'D_rev', 'B_sph', 'C_uni', 'ground', 'l1', 'l2', 'l3', 'l1.y_actuator'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['A_rev']=joints['A_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['D_rev']=joints['D_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['B_sph']=joints['B_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['C_uni']=joints['C_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['ground']=bodies['ground'].acc_rhs(qdot)
	 acc_rhs_s['l1']=bodies['l1'].acc_rhs(qdot)
	 acc_rhs_s['l2']=bodies['l2'].acc_rhs(qdot)
	 acc_rhs_s['l3']=bodies['l3'].acc_rhs(qdot)
	 acc_rhs_s['l1.y_actuator']=actuators['l1.y_actuator'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((9,1)),format='csc') 
	 return system.A.reshape((28,)) 


