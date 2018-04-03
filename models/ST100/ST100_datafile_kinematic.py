import numpy as np 
import pandas as pd 
from scipy import sparse 


jac_df=pd.DataFrame([8 *[None]],columns=['chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie'],index=['ucaf_rev', 'lcaf_rev', 'wc_rev', 'tro_sph', 'ucao_sph', 'lcao_sph', 'd_m_cyl', 'strut_lca_uni', 'strut_ch_uni', 'tri_uni', 'chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie', 'wc_rev_actuator', 'wheel.z_actuator'])
def cq(q,bodies,joints,actuators): 
	 jac_df.loc['ucaf_rev','uca']=joints['ucaf_rev'].jacobian_i(q)
	 jac_df.loc['ucaf_rev','chassis']=joints['ucaf_rev'].jacobian_j(q)
	 jac_df.loc['lcaf_rev','lca']=joints['lcaf_rev'].jacobian_i(q)
	 jac_df.loc['lcaf_rev','chassis']=joints['lcaf_rev'].jacobian_j(q)
	 jac_df.loc['wc_rev','wheel']=joints['wc_rev'].jacobian_i(q)
	 jac_df.loc['wc_rev','upright']=joints['wc_rev'].jacobian_j(q)
	 jac_df.loc['tro_sph','tie']=joints['tro_sph'].jacobian_i(q)
	 jac_df.loc['tro_sph','upright']=joints['tro_sph'].jacobian_j(q)
	 jac_df.loc['ucao_sph','uca']=joints['ucao_sph'].jacobian_i(q)
	 jac_df.loc['ucao_sph','upright']=joints['ucao_sph'].jacobian_j(q)
	 jac_df.loc['lcao_sph','lca']=joints['lcao_sph'].jacobian_i(q)
	 jac_df.loc['lcao_sph','upright']=joints['lcao_sph'].jacobian_j(q)
	 jac_df.loc['d_m_cyl','strut_upper']=joints['d_m_cyl'].jacobian_i(q)
	 jac_df.loc['d_m_cyl','strut_lower']=joints['d_m_cyl'].jacobian_j(q)
	 jac_df.loc['strut_lca_uni','lca']=joints['strut_lca_uni'].jacobian_i(q)
	 jac_df.loc['strut_lca_uni','strut_upper']=joints['strut_lca_uni'].jacobian_j(q)
	 jac_df.loc['strut_ch_uni','strut_lower']=joints['strut_ch_uni'].jacobian_i(q)
	 jac_df.loc['strut_ch_uni','chassis']=joints['strut_ch_uni'].jacobian_j(q)
	 jac_df.loc['tri_uni','chassis']=joints['tri_uni'].jacobian_i(q)
	 jac_df.loc['tri_uni','tie']=joints['tri_uni'].jacobian_j(q)
	 jac_df.loc['chassis','chassis']=bodies['chassis'].mount_jacobian(q)
	 jac_df.loc['uca','uca']=bodies['uca'].unity_jacobian(q)
	 jac_df.loc['lca','lca']=bodies['lca'].unity_jacobian(q)
	 jac_df.loc['upright','upright']=bodies['upright'].unity_jacobian(q)
	 jac_df.loc['wheel','wheel']=bodies['wheel'].unity_jacobian(q)
	 jac_df.loc['strut_lower','strut_lower']=bodies['strut_lower'].unity_jacobian(q)
	 jac_df.loc['strut_upper','strut_upper']=bodies['strut_upper'].unity_jacobian(q)
	 jac_df.loc['tie','tie']=bodies['tie'].unity_jacobian(q)
	 jac_df.loc['wc_rev_actuator','wheel']=actuators['wc_rev_actuator'].jacobian_i(q)
	 jac_df.loc['wc_rev_actuator','upright']=actuators['wc_rev_actuator'].jacobian_j(q)
	 jac_df.loc['wheel.z_actuator','wheel']=actuators['wheel.z_actuator'].jacobian()
	 jac=sparse.bmat(jac_df,format='csc') 
	 return jac 


eq_s=pd.Series([20 *[None]],index=['ucaf_rev', 'lcaf_rev', 'wc_rev', 'tro_sph', 'ucao_sph', 'lcao_sph', 'd_m_cyl', 'strut_lca_uni', 'strut_ch_uni', 'tri_uni', 'chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie', 'wc_rev_actuator', 'wheel.z_actuator'])
def eq(q,bodies,joints,actuators): 
	 eq_s['ucaf_rev']=joints['ucaf_rev'].equations(q)
	 eq_s['lcaf_rev']=joints['lcaf_rev'].equations(q)
	 eq_s['wc_rev']=joints['wc_rev'].equations(q)
	 eq_s['tro_sph']=joints['tro_sph'].equations(q)
	 eq_s['ucao_sph']=joints['ucao_sph'].equations(q)
	 eq_s['lcao_sph']=joints['lcao_sph'].equations(q)
	 eq_s['d_m_cyl']=joints['d_m_cyl'].equations(q)
	 eq_s['strut_lca_uni']=joints['strut_lca_uni'].equations(q)
	 eq_s['strut_ch_uni']=joints['strut_ch_uni'].equations(q)
	 eq_s['tri_uni']=joints['tri_uni'].equations(q)
	 eq_s['chassis']=bodies['chassis'].mount_equation(q)
	 eq_s['uca']=bodies['uca'].unity_equation(q)
	 eq_s['lca']=bodies['lca'].unity_equation(q)
	 eq_s['upright']=bodies['upright'].unity_equation(q)
	 eq_s['wheel']=bodies['wheel'].unity_equation(q)
	 eq_s['strut_lower']=bodies['strut_lower'].unity_equation(q)
	 eq_s['strut_upper']=bodies['strut_upper'].unity_equation(q)
	 eq_s['tie']=bodies['tie'].unity_equation(q)
	 eq_s['wc_rev_actuator']=actuators['wc_rev_actuator'].equations(q)
	 eq_s['wheel.z_actuator']=actuators['wheel.z_actuator'].equations(q)
	 system=sparse.bmat(eq_s.values.reshape((20,1)),format='csc') 
	 return system.A.reshape((56,)) 


vel_rhs_s=pd.Series([20 *[None]],index=['ucaf_rev', 'lcaf_rev', 'wc_rev', 'tro_sph', 'ucao_sph', 'lcao_sph', 'd_m_cyl', 'strut_lca_uni', 'strut_ch_uni', 'tri_uni', 'chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie', 'wc_rev_actuator', 'wheel.z_actuator'])
def vel_rhs(actuators): 
	 vrhs=np.zeros((54,1))
	 vrhs=np.concatenate([vrhs,actuators['wc_rev_actuator'].vel_rhs()]) 
	 vrhs=np.concatenate([vrhs,actuators['wheel.z_actuator'].vel_rhs()]) 
	 return vrhs 
acc_rhs_s=pd.Series([20 *[None]],index=['ucaf_rev', 'lcaf_rev', 'wc_rev', 'tro_sph', 'ucao_sph', 'lcao_sph', 'd_m_cyl', 'strut_lca_uni', 'strut_ch_uni', 'tri_uni', 'chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie', 'wc_rev_actuator', 'wheel.z_actuator'])
def acc_rhs(q,qdot,bodies,joints,actuators): 
	 acc_rhs_s['ucaf_rev']=joints['ucaf_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['lcaf_rev']=joints['lcaf_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['wc_rev']=joints['wc_rev'].acc_rhs(q,qdot)
	 acc_rhs_s['tro_sph']=joints['tro_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['ucao_sph']=joints['ucao_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['lcao_sph']=joints['lcao_sph'].acc_rhs(q,qdot)
	 acc_rhs_s['d_m_cyl']=joints['d_m_cyl'].acc_rhs(q,qdot)
	 acc_rhs_s['strut_lca_uni']=joints['strut_lca_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['strut_ch_uni']=joints['strut_ch_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['tri_uni']=joints['tri_uni'].acc_rhs(q,qdot)
	 acc_rhs_s['chassis']=bodies['chassis'].mount_acc_rhs(qdot)
	 acc_rhs_s['uca']=bodies['uca'].acc_rhs(qdot)
	 acc_rhs_s['lca']=bodies['lca'].acc_rhs(qdot)
	 acc_rhs_s['upright']=bodies['upright'].acc_rhs(qdot)
	 acc_rhs_s['wheel']=bodies['wheel'].acc_rhs(qdot)
	 acc_rhs_s['strut_lower']=bodies['strut_lower'].acc_rhs(qdot)
	 acc_rhs_s['strut_upper']=bodies['strut_upper'].acc_rhs(qdot)
	 acc_rhs_s['tie']=bodies['tie'].acc_rhs(qdot)
	 acc_rhs_s['wc_rev_actuator']=actuators['wc_rev_actuator'].acc_rhs(q,qdot)
	 acc_rhs_s['wheel.z_actuator']=actuators['wheel.z_actuator'].acc_rhs(q,qdot)
	 system=sparse.bmat(acc_rhs_s.values.reshape((20,1)),format='csc') 
	 return system.A.reshape((56,)) 


mass_matrix_df=pd.DataFrame([8 *[None]],columns=['chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie'],index=['chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie'])
def mass_matrix(q,bodies): 
	 mass_matrix_df.loc['chassis','chassis']=bodies['chassis'].mass_matrix(q)
	 mass_matrix_df.loc['uca','uca']=bodies['uca'].mass_matrix(q)
	 mass_matrix_df.loc['lca','lca']=bodies['lca'].mass_matrix(q)
	 mass_matrix_df.loc['upright','upright']=bodies['upright'].mass_matrix(q)
	 mass_matrix_df.loc['wheel','wheel']=bodies['wheel'].mass_matrix(q)
	 mass_matrix_df.loc['strut_lower','strut_lower']=bodies['strut_lower'].mass_matrix(q)
	 mass_matrix_df.loc['strut_upper','strut_upper']=bodies['strut_upper'].mass_matrix(q)
	 mass_matrix_df.loc['tie','tie']=bodies['tie'].mass_matrix(q)
	 mass_matrix=sparse.bmat(mass_matrix_df,format='csc') 
	 return mass_matrix 


Qg_s=pd.Series([8 *np.zeros((7,1))],index=['chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie'])
def Qg(bodies): 
	 Qg_s['chassis']=bodies['chassis'].gravity()
	 Qg_s['uca']=bodies['uca'].gravity()
	 Qg_s['lca']=bodies['lca'].gravity()
	 Qg_s['upright']=bodies['upright'].gravity()
	 Qg_s['wheel']=bodies['wheel'].gravity()
	 Qg_s['strut_lower']=bodies['strut_lower'].gravity()
	 Qg_s['strut_upper']=bodies['strut_upper'].gravity()
	 Qg_s['tie']=bodies['tie'].gravity()
	 system=sparse.bmat(Qg_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((56,)) 


Qv_s=pd.Series([8 *np.zeros((7,1))],index=['chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie'])
def Qv(bodies,q,qdot): 
	 Qv_s['chassis']=bodies['chassis'].centrifugal(q,qdot)
	 Qv_s['uca']=bodies['uca'].centrifugal(q,qdot)
	 Qv_s['lca']=bodies['lca'].centrifugal(q,qdot)
	 Qv_s['upright']=bodies['upright'].centrifugal(q,qdot)
	 Qv_s['wheel']=bodies['wheel'].centrifugal(q,qdot)
	 Qv_s['strut_lower']=bodies['strut_lower'].centrifugal(q,qdot)
	 Qv_s['strut_upper']=bodies['strut_upper'].centrifugal(q,qdot)
	 Qv_s['tie']=bodies['tie'].centrifugal(q,qdot)
	 system=sparse.bmat(Qv_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((56,)) 


Qa_s=pd.Series([8 *np.zeros((7,1))],index=['chassis', 'uca', 'lca', 'upright', 'wheel', 'strut_lower', 'strut_upper', 'tie'])
def Qa(forces,q,qdot): 
	 Qi,Qj=forces['gk_w11'].equation(q,qdot) 
	 Qa_s['strut_lower']=Qi
	 Qa_s['strut_upper']=Qj
	 system=sparse.bmat(Qa_s.values.reshape((8,1)),format='csc') 
	 return system.A.reshape((56,)) 


JR_s=pd.Series(np.zeros((60)),index=['ucaf_rev_Fx', 'ucaf_rev_Fy', 'ucaf_rev_Fz', 'ucaf_rev_Mx', 'ucaf_rev_My', 'ucaf_rev_Mz', 'lcaf_rev_Fx', 'lcaf_rev_Fy', 'lcaf_rev_Fz', 'lcaf_rev_Mx', 'lcaf_rev_My', 'lcaf_rev_Mz', 'wc_rev_Fx', 'wc_rev_Fy', 'wc_rev_Fz', 'wc_rev_Mx', 'wc_rev_My', 'wc_rev_Mz', 'tro_sph_Fx', 'tro_sph_Fy', 'tro_sph_Fz', 'tro_sph_Mx', 'tro_sph_My', 'tro_sph_Mz', 'ucao_sph_Fx', 'ucao_sph_Fy', 'ucao_sph_Fz', 'ucao_sph_Mx', 'ucao_sph_My', 'ucao_sph_Mz', 'lcao_sph_Fx', 'lcao_sph_Fy', 'lcao_sph_Fz', 'lcao_sph_Mx', 'lcao_sph_My', 'lcao_sph_Mz', 'd_m_cyl_Fx', 'd_m_cyl_Fy', 'd_m_cyl_Fz', 'd_m_cyl_Mx', 'd_m_cyl_My', 'd_m_cyl_Mz', 'strut_lca_uni_Fx', 'strut_lca_uni_Fy', 'strut_lca_uni_Fz', 'strut_lca_uni_Mx', 'strut_lca_uni_My', 'strut_lca_uni_Mz', 'strut_ch_uni_Fx', 'strut_ch_uni_Fy', 'strut_ch_uni_Fz', 'strut_ch_uni_Mx', 'strut_ch_uni_My', 'strut_ch_uni_Mz', 'tri_uni_Fx', 'tri_uni_Fy', 'tri_uni_Fz', 'tri_uni_Mx', 'tri_uni_My', 'tri_uni_Mz'])
def JR(joints,q,lamda): 
	 JR_s[['ucaf_rev_Fx', 'ucaf_rev_Fy', 'ucaf_rev_Fz', 'ucaf_rev_Mx', 'ucaf_rev_My', 'ucaf_rev_Mz']]=joints['ucaf_rev'].reactions(q,lamda)
	 JR_s[['lcaf_rev_Fx', 'lcaf_rev_Fy', 'lcaf_rev_Fz', 'lcaf_rev_Mx', 'lcaf_rev_My', 'lcaf_rev_Mz']]=joints['lcaf_rev'].reactions(q,lamda)
	 JR_s[['wc_rev_Fx', 'wc_rev_Fy', 'wc_rev_Fz', 'wc_rev_Mx', 'wc_rev_My', 'wc_rev_Mz']]=joints['wc_rev'].reactions(q,lamda)
	 JR_s[['tro_sph_Fx', 'tro_sph_Fy', 'tro_sph_Fz', 'tro_sph_Mx', 'tro_sph_My', 'tro_sph_Mz']]=joints['tro_sph'].reactions(q,lamda)
	 JR_s[['ucao_sph_Fx', 'ucao_sph_Fy', 'ucao_sph_Fz', 'ucao_sph_Mx', 'ucao_sph_My', 'ucao_sph_Mz']]=joints['ucao_sph'].reactions(q,lamda)
	 JR_s[['lcao_sph_Fx', 'lcao_sph_Fy', 'lcao_sph_Fz', 'lcao_sph_Mx', 'lcao_sph_My', 'lcao_sph_Mz']]=joints['lcao_sph'].reactions(q,lamda)
	 JR_s[['d_m_cyl_Fx', 'd_m_cyl_Fy', 'd_m_cyl_Fz', 'd_m_cyl_Mx', 'd_m_cyl_My', 'd_m_cyl_Mz']]=joints['d_m_cyl'].reactions(q,lamda)
	 JR_s[['strut_lca_uni_Fx', 'strut_lca_uni_Fy', 'strut_lca_uni_Fz', 'strut_lca_uni_Mx', 'strut_lca_uni_My', 'strut_lca_uni_Mz']]=joints['strut_lca_uni'].reactions(q,lamda)
	 JR_s[['strut_ch_uni_Fx', 'strut_ch_uni_Fy', 'strut_ch_uni_Fz', 'strut_ch_uni_Mx', 'strut_ch_uni_My', 'strut_ch_uni_Mz']]=joints['strut_ch_uni'].reactions(q,lamda)
	 JR_s[['tri_uni_Fx', 'tri_uni_Fy', 'tri_uni_Fz', 'tri_uni_Mx', 'tri_uni_My', 'tri_uni_Mz']]=joints['tri_uni'].reactions(q,lamda)
	 return JR_s 


