# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:08:03 2017

@author: khale
"""

import numpy as np
import pandas as pd
from base import vector, ep2dcm, B, orient_along_axis, vec2skew, E
from scipy import sparse
import scipy as sc


def acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v2,Aj,Bjv2,Hjv2,bjd):
        
    rhs=-2*np.linalg.multi_dot([bid.T,Biv1.T,Bjv2,bjd])-\
            np.linalg.multi_dot([v2.T,Aj.T,Hiv1,bid])-\
            np.linalg.multi_dot([v1.T,Ai.T,Hjv2,bjd])
            
    return rhs


def acc_dp2_rhs(v1,Ai,Biv1,Hiv1,bid,rij,Hip,Hjp,bjd,rij_dot):

    
    rhs=-2*np.linalg.multi_dot([bid.T,Biv1.T,rij_dot])-\
        np.linalg.multi_dot([rij.T,Hiv1,bid])-\
        np.linalg.multi_dot([v1.T,Ai.T,Hip,bid])+\
        np.linalg.multi_dot([v1.T,Ai.T,Hjp,bjd])
    
    return rhs


def acc_sph_rhs(betai_dot,Hip,Hjp,betaj_dot):    
    
    H  = np.concatenate((Hip,-Hjp),axis=1)
    qd = np.concatenate((betai_dot,betaj_dot)).reshape((8,1))
    
    return -H.dot(qd)    



class joint(object):
    def __init__(self,location,i_body,j_body,axis=[0,0,1]):
        
        # defining the basic attributes in the joint class
        self.name=location.name
        self.i_body=i_body
        self.j_body=j_body
        self.type="None" # for representing the joint type in subclasses
        self.typ =location.typ
        
        self._loc=location
        # defining the joint axis which is needed for the different types of joints
        self.axis=axis
        
        self.frame=orient_along_axis(axis)
        self.u_irf=i_body.dcm.T.dot(self.frame)
        self.u_jrf=j_body.dcm.T.dot(self.frame)
        
        # joint locations wrt to each body origin cm referenced to the body reference
        self.u_i=i_body.dcm.T.dot(self._loc-i_body.R)
        self.u_j=j_body.dcm.T.dot(self._loc-j_body.R)
        
        # axes to be used in the cylindrical and revolute joints classes
        self.vii=vector(self.u_irf[:,0])
        self.vij=vector(self.u_irf[:,1])
        self.vik=vector(self.u_irf[:,2])
        
        self.vji=vector(self.u_jrf[:,0])
        self.vjj=vector(self.u_jrf[:,1])
        self.vjk=vector(self.u_jrf[:,2])
        
#        location.body=i_body
        
        
    @property
    def dic(self):
        name=self.name+'.'
        loc=self.loc
        return {name+'x':loc.x,name+'y':loc.y,name+'z':loc.z}
        
    def joint_pos(self,q):
        qi=q[self.i_body.dic.index]
        Ri=vector(qi[0:3]).a
        Ai=ep2dcm(qi[3:])
        ui=self.u_i
        r_p=Ri+Ai.dot(ui)
        return r_p
    
    
    def reactions(self,q,lamda):
        l=lamda[self.index].values.reshape((self.nc,1))
        betai=q[self.i_body.dic.index][3:]
        Ai=ep2dcm(betai)
        ui=vector(Ai.dot(self.u_i))
        
        jac  = self.jacobian_i(q)
        jacR = jac[:,:3]
        jacP = jac[:,3:]
        
        F=-jacR.T.dot(l)
        T=-jacP.T.dot(l)
        cartesian_moment=0.5*E(betai).dot(T)
        joint_torque=0.5*E(betai).dot(T)-vec2skew(ui).dot(F)
        
#        print('Force   = %s'%F)
#        print('Cmoment = %s'%cartesian_moment)
#        print('FTorque = %s'%vec2skew(ui).dot(F))
#        print('JTorque = %s'%joint_torque)
                
        return np.concatenate([[F],[joint_torque]]).reshape((6,))
        
    
    @property
    def reaction_index(self):
        name=self.name
        indices=[name+'_%s'%i for i in ['Fx','Fy','Fz','Mx','My','Mz']]
        return indices
    
    @property
    def index(self):
        '''
        index for joint lagrange multipliers based on number of constraints
        '''
        name=self.name
        indices=[name+'_eq%s'%i for i in range(self.nc)]
        return indices

#    def __repr__(self):
#        return self.type+' at '+ str(self.loc.row)+' connecting '\
#                +self.i_body.name +' & '+ self.j_body.name



class spherical(joint):
    '''
    Defining the at point constraint which constraints the relative translations
    between two bodies at a given point 
    '''
    def __init__(self,location,i_body,j_body,axis=[0,0,1]):
        super().__init__(location,i_body,j_body,axis)
        
        self.type='spherical joint'
        self.name=self.name+'_sph'
        self.nc=3

    
       
    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
        
        ui=self.u_i
        uj=self.u_j
        
        r_p=Ri+Ai.dot(ui)-Rj-Aj.dot(uj)
        return r_p
    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        
        betai=qi[3:]
        Hp = B(betai,self.u_i)
        I  = sparse.eye(3,format='csr')
        
        jac = sparse.bmat([[I,Hp]],format='csr')
        return jac
    
    def jacobian_j(self,q):
        
        qj=q[self.j_body.dic.index]
        
        betaj=qj[3:]
        Hp = B(betaj,self.u_j)
        I  = sparse.eye(3,format='csr')
        
        jac = sparse.bmat([[-I,-Hp]],format='csr')
        return jac
    
    def acc_rhs(self,q,qdot):
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        Hip=B(betai_dot,self.u_i)
        Hjp=B(betaj_dot,self.u_j)
        
        return acc_sph_rhs(betai_dot,Hip,Hjp,betaj_dot)
        

    def mir(location,i_body,j_body,axis=[0,0,1]):
        
        loc_l, loc_r = location
        ibody_l, ibody_r = i_body
        jbody_l, jbody_r = j_body
        
        left  = spherical(loc_l,ibody_l,jbody_l)
        right = spherical(loc_r,ibody_r,jbody_r)
        
        return pd.Series([left,right],index=['l','r'])
        

class cylindrical(joint):
    def __init__(self,location,i_body,j_body,axis):
        super().__init__(location,i_body,j_body,axis)
        self.type='cylindrical joint'
        self.name=self.name+'_cyl'
        self.nc=4
        self.p2=vector(location)+10*vector(axis).unit
        self.u_j=j_body.dcm.T.dot(self.p2-j_body.R)
        


    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)+10*v3

        
        eq1=np.linalg.multi_dot([v1.T,Ai.T,Aj,v3])
        eq2=np.linalg.multi_dot([v2.T,Ai.T,Aj,v3])
        eq3=np.linalg.multi_dot([v1.T,Ai.T,rij])
        eq4=np.linalg.multi_dot([v2.T,Ai.T,rij])
        
        
        c=[eq1,eq2,eq3,eq4]
        return np.array([c]).reshape((4,1))
    
    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        v3=Aj.dot(self.vjk)
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)+10*v3
        
        Hiv1=B(betai,self.vii)
        Hiv2=B(betai,self.vij)
        Hiup=B(betai,self.u_i)
        
        Z=sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[Z,v3.T.dot(Hiv1)],
                         [Z,v3.T.dot(Hiv2)],
                         [v1.T,rij.T.dot(Hiv1)+v1.T.dot(Hiup)],
                         [v2.T,rij.T.dot(Hiv2)+v2.T.dot(Hiup)]],format='csr')
        
        return jac
    
    def jacobian_j(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ai=ep2dcm(betai)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        
        Hjv3=B(betaj,self.vjk)
        Hjup=B(betaj,self.u_j)
        
        Z=sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[  Z,   v1.T.dot(Hjv3)],
                         [  Z,   v2.T.dot(Hjv3)],
                         [-v1.T, -v1.T.dot(Hjup)],
                         [-v2.T, -v2.T.dot(Hjup)]],format='csr')
        
        return jac
    
    
    def acc_rhs(self,q,qdot):
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        Ri=qi[0:3].values.reshape((3,1))
        Rj=qj[0:3].values.reshape((3,1))
        betai=qi[3:]
        betaj=qj[3:]
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)

        Ri_dot=qi_dot[0:3].values.reshape((3,1))
        Rj_dot=qj_dot[0:3].values.reshape((3,1))
        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        bid=betai_dot.values.reshape((4,1))
        bjd=betaj_dot.values.reshape((4,1))
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)

        
        Biv1=B(betai,v1)
        Biv2=B(betai,v2)
        Bjv3=B(betaj,v3)
        Bip=B(betai,self.u_i)
        Bjp=B(betaj,self.u_j)


        Hiv1=B(betai_dot,v1)
        Hiv2=B(betai_dot,v2)
        Hjv3=B(betaj_dot,v3)
        Hip =B(betai_dot,self.u_i)
        Hjp =B(betaj_dot,self.u_j)

        
        rij_dot=Ri_dot+Bip.dot(bid)-Rj_dot-Bjp.dot(bjd)
        
        rhs1=acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v3,Aj,Bjv3,Hjv3,bjd)
        rhs2=acc_dp1_rhs(v2,Ai,Biv2,Hiv2,bid,v3,Aj,Bjv3,Hjv3,bjd)
        rhs3=acc_dp2_rhs(v1,Ai,Biv1,Hiv1,bid,rij,Hip,Hjp,bjd,rij_dot)
        rhs4=acc_dp2_rhs(v2,Ai,Biv2,Hiv2,bid,rij,Hip,Hjp,bjd,rij_dot)
        
        return np.concatenate([rhs1,rhs2,rhs3,rhs4])
    

    
    def mir(location,i_body,j_body,axis):
        
        loc_l, loc_r = location
        ibody_l, ibody_r = i_body
        jbody_l, jbody_r = j_body
        
        ax_l, ax_r = axis
        
        left  = cylindrical(loc_l,ibody_l,jbody_l,ax_l)
        right = cylindrical(loc_r,ibody_r,jbody_r,ax_r)
        
        return pd.Series([left,right],index=['l','r'])



class translational(joint):
    def __init__(self,location,i_body,j_body,axis):
        super().__init__(location,i_body,j_body,axis)
        self.type='translational joint'
        self.name=self.name+'_trans'
        self.nc=5
        self.p2=vector(location)+10*vector(axis).unit
        self.u_j=j_body.dcm.T.dot(self.p2-j_body.R)
        


    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        v4=self.vjj
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)+10*v3

        
        eq1=np.linalg.multi_dot([v1.T,Ai.T,Aj,v3])
        eq2=np.linalg.multi_dot([v2.T,Ai.T,Aj,v3])
        eq3=np.linalg.multi_dot([v1.T,Ai.T,rij])
        eq4=np.linalg.multi_dot([v2.T,Ai.T,rij])
        eq5=np.linalg.multi_dot([v1.T,Ai.T,Aj,v4])
        
        
        c=[eq1,eq2,eq3,eq4,eq5]
        return np.array([c]).reshape((5,1))
    
    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        v3=Aj.dot(self.vjk)
        v4=Aj.dot(self.vjj)
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)+10*v3
        
        Hiv1=B(betai,self.vii)
        Hiv2=B(betai,self.vij)
        Hiup=B(betai,self.u_i)
        
        Z=sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[Z,v3.T.dot(Hiv1)],
                         [Z,v3.T.dot(Hiv2)],
                         [v1.T,rij.T.dot(Hiv1)+v1.T.dot(Hiup)],
                         [v2.T,rij.T.dot(Hiv2)+v2.T.dot(Hiup)],
                         [Z,v4.T.dot(Hiv1)]],format='csr')
        
        return jac
    
    def jacobian_j(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ai=ep2dcm(betai)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        
        Hjv3=B(betaj,self.vjk)
        Hjv4=B(betaj,self.vjj)
        Hjup=B(betaj,self.u_j)
        
        Z=sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[  Z,   v1.T.dot(Hjv3)],
                         [  Z,   v2.T.dot(Hjv3)],
                         [-v1.T, -v1.T.dot(Hjup)],
                         [-v2.T, -v2.T.dot(Hjup)],
                         [  Z,   v1.T.dot(Hjv4)]],format='csr')
        
        return jac
    
    
    def acc_rhs(self,q,qdot):
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        Ri=qi[0:3].values.reshape((3,1))
        Rj=qj[0:3].values.reshape((3,1))
        betai=qi[3:]
        betaj=qj[3:]
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)

        Ri_dot=qi_dot[0:3].values.reshape((3,1))
        Rj_dot=qj_dot[0:3].values.reshape((3,1))
        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        bid=betai_dot.values.reshape((4,1))
        bjd=betaj_dot.values.reshape((4,1))
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        v4=self.vjj
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)

        
        Biv1=B(betai,v1)
        Biv2=B(betai,v2)
        Bjv3=B(betaj,v3)
        Bjv4=B(betaj,v4)
        Bip=B(betai,self.u_i)
        Bjp=B(betaj,self.u_j)


        Hiv1=B(betai_dot,v1)
        Hiv2=B(betai_dot,v2)
        Hjv3=B(betaj_dot,v3)
        Hjv4=B(betaj_dot,v4)
        Hip =B(betai_dot,self.u_i)
        Hjp =B(betaj_dot,self.u_j)

        
        rij_dot=Ri_dot+Bip.dot(bid)-Rj_dot-Bjp.dot(bjd)
        
        rhs1=acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v3,Aj,Bjv3,Hjv3,bjd)
        rhs2=acc_dp1_rhs(v2,Ai,Biv2,Hiv2,bid,v3,Aj,Bjv3,Hjv3,bjd)
        rhs3=acc_dp2_rhs(v1,Ai,Biv1,Hiv1,bid,rij,Hip,Hjp,bjd,rij_dot)
        rhs4=acc_dp2_rhs(v2,Ai,Biv2,Hiv2,bid,rij,Hip,Hjp,bjd,rij_dot)
        rhs5=acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v4,Aj,Bjv4,Hjv4,bjd)
        
        return np.concatenate([rhs1,rhs2,rhs3,rhs4,rhs5])
    

    
    def mir(location,i_body,j_body,axis):
        
        loc_l, loc_r = location
        ibody_l, ibody_r = i_body
        jbody_l, jbody_r = j_body
        
        ax_l, ax_r = axis
        
        left  = cylindrical(loc_l,ibody_l,jbody_l,ax_l)
        right = cylindrical(loc_r,ibody_r,jbody_r,ax_r)
        
        return pd.Series([left,right],index=['l','r'])



class revolute(joint):
    def __init__(self,location,i_body,j_body,axis):
        super().__init__(location,i_body,j_body,axis)
        self.type='revolute joint'
        self.name=self.name+'_rev'
        self.nc=5
        
    
    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        v3=Aj.dot(self.vjk)
        
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)
        
        eq1,eq2,eq3=rij
        
        eq4=v1.T.dot(v3)
        eq5=v2.T.dot(v3)
    
        
        c=[float(i) for i in (eq1,eq2,eq3,eq4,eq5)]
        return np.array([c]).reshape((5,1))
    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Aj=ep2dcm(betaj)
        
        v3=Aj.dot(self.vjk)
        
        I    = sparse.eye(3,format='csr')
        Hiup = B(betai,self.u_i)
        Hiv1 = B(betai,self.vii)
        Hiv2 = B(betai,self.vij)
        Z    = sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[I,Hiup],
                         [Z,v3.T.dot(Hiv1)],
                         [Z,v3.T.dot(Hiv2)]],format='csr')
        return jac
    
    def jacobian_j(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ai=ep2dcm(betai)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        
        I    = sparse.eye(3,format='csr')
        Hjup = B(betaj,self.u_j)
        Hjv3 = B(betaj,self.vjk)
        Hjv3 = B(betaj,self.vjk)
        Z    = sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[-I,-Hjup],
                         [Z,v1.T.dot(Hjv3)],
                         [Z,v2.T.dot(Hjv3)]],format='csr')
        return jac
    
    
    def acc_rhs(self,q,qdot):
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)

        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        bid=betai_dot.values.reshape((4,1))
        bjd=betaj_dot.values.reshape((4,1))
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        
        Biv1=B(betai,v1)
        Biv2=B(betai,v2)
        Bjv3=B(betaj,v3)


        Hiv1=B(betai_dot,v1)
        Hiv2=B(betai_dot,v2)
        Hjv3=B(betaj_dot,v3)
        Hip =B(betai_dot,self.u_i)
        Hjp =B(betaj_dot,self.u_j)

                
        rhs123=acc_sph_rhs(betai_dot,Hip,Hjp,betaj_dot)
        rhs4  =acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v3,Aj,Bjv3,Hjv3,bjd)
        rhs5  =acc_dp1_rhs(v2,Ai,Biv2,Hiv2,bid,v3,Aj,Bjv3,Hjv3,bjd)
        
        return np.concatenate([rhs123,rhs4,rhs5])
    
    
    def mir(location,i_body,j_body,axis):
        
        loc_l, loc_r = location
        ibody_l, ibody_r = i_body
        jbody_l, jbody_r = j_body
        
        ax_l, ax_r = axis
        
        left  = revolute(loc_l,ibody_l,jbody_l,ax_l)
        right = revolute(loc_r,ibody_r,jbody_r,ax_r)
        
        return pd.Series([left,right],index=['l','r'])


class universal(joint):
    def __init__(self,location,i_body,j_body,i_rot,j_rot):
        super().__init__(location,i_body,j_body)
        self.type='universal joint'
        self.name=self.name+'_uni'
        self.nc=4
        
        self.i_rot=vector(i_rot)
        self.j_rot=vector(j_rot)
        
        self.i_rot_i = self.i_body.dcm.T.dot(self.i_rot)
        self.j_rot_j = self.j_body.dcm.T.dot(self.j_rot)
        
        
        self.angle = np.sin(np.radians(vector(self.i_rot).angle_between(vector(self.j_rot))))

        
        if abs(self.angle)<=1e-7:
            # collinear axis 
            self.u_irf=orient_along_axis(self.i_rot_i)
            jax_j=j_body.dcm.T.dot(i_body.dcm.dot(self.u_irf[:,0]))
            self.u_jrf=orient_along_axis(self.j_rot_j,vector(jax_j))
            self.h_i=vector(self.u_irf[:,1]).a
            self.h_j=vector(self.u_jrf[:,0]).a
            
        else:
            # Angled axis
            #print('angled')
            ax=(vector(i_rot).cross(vector(j_rot))).unit
            ax_i=i_body.dcm.T.dot(ax)
            ax_j=j_body.dcm.T.dot(ax)
            
            self.u_irf=orient_along_axis(self.i_rot_i,i_vector=ax_i)
            self.u_jrf=orient_along_axis(self.j_rot_j,i_vector=ax_j)
            
            self.h_i=vector(self.u_irf[:,1]).a
            self.h_j=vector(self.u_jrf[:,0]).a
        
    
    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
    
        eq1,eq2,eq3=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)
        eq4=np.linalg.multi_dot([self.h_i.T,Ai.T,Aj,self.h_j])
        
        c=[float(i) for i in (eq1,eq2,eq3,eq4)]
        return np.array([c]).reshape((4,1))

    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Aj=ep2dcm(betaj)
        
        h2=Aj.dot(self.h_j)
        
        I    = sparse.eye(3,format='csr')
        Hiup = B(betai,self.u_i)
        Hih1 = B(betai,self.h_i)
        Z    = sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[I,Hiup],
                         [Z,h2.T.dot(Hih1)]],format='csr')
        return jac
    
    def jacobian_j(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ai=ep2dcm(betai)
        
        h1=Ai.dot(self.h_i)
        
        I    = sparse.eye(3,format='csr')
        Hjup = B(betaj,self.u_j)
        Hjh2 = B(betaj,self.h_j)
        Z    = sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[-I,-Hjup],
                         [Z,h1.T.dot(Hjh2)]],format='csr')
        return jac
    
    def acc_rhs(self,q,qdot):
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)

        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        bid=betai_dot.values.reshape((4,1))
        bjd=betaj_dot.values.reshape((4,1))
        
        h1=self.h_i
        h2=self.h_j
        
        Bih1=B(betai,h1)
        Bjh2=B(betaj,h2)


        Hih1=B(betai_dot,h1)
        Hjh2=B(betaj_dot,h2)
        Hip =B(betai_dot,self.u_i)
        Hjp =B(betaj_dot,self.u_j)

                
        rhs123=acc_sph_rhs(betai_dot,Hip,Hjp,betaj_dot)
        rhs4  =acc_dp1_rhs(h1,Ai,Bih1,Hih1,bid,h2,Aj,Bjh2,Hjh2,bjd)
        
        return np.concatenate([rhs123,rhs4])
    
    
    def mir(location,i_body,j_body,i_rot,j_rot):
        
        loc_l, loc_r = location
        ibody_l, ibody_r = i_body
        jbody_l, jbody_r = j_body
        
        irot_l, irot_r = i_rot
        jrot_l, jrot_r = j_rot
        
        left  = universal(loc_l,ibody_l,jbody_l,irot_l,jrot_l)
        right = universal(loc_r,ibody_r,jbody_r,irot_r,jrot_r)
        
        return pd.Series([left,right],index=['l','r'])
    
class bounce_roll(joint):
    def __init__(self,location,i_body,j_body,bounce_ax,roll_ax):
        super().__init__(location,i_body,j_body,bounce_ax)
        self.type='bounce_roll joint'
        self.name=self.name+'_br'
        self.nc=4
        self.p2=vector(location)+10*vector(bounce_ax).unit
        self.u_j=j_body.dcm.T.dot(self.p2-j_body.R)
        
        self.frame=orient_along_axis(bounce_ax,roll_ax)
        self.u_irf=i_body.dcm.T.dot(self.frame)
        self.u_jrf=j_body.dcm.T.dot(self.frame)
        
        self.vii=vector(self.u_irf[:,0])
        self.vij=vector(self.u_irf[:,1])
        self.vik=vector(self.u_irf[:,2])
        
        self.vji=vector(self.u_jrf[:,0])
        self.vjj=vector(self.u_jrf[:,1])
        self.vjk=vector(self.u_jrf[:,2])
        


    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        v4=self.vjj
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)+10*v3

        
        eq1=np.linalg.multi_dot([v1.T,Ai.T,Aj,v3])
        eq3=np.linalg.multi_dot([v1.T,Ai.T,rij])
        eq4=np.linalg.multi_dot([v2.T,Ai.T,rij])
        eq5=np.linalg.multi_dot([v1.T,Ai.T,Aj,v4])
        
        
        c=[eq1,eq3,eq4,eq5]
        return np.array([c]).reshape((4,1))
    
    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ri=vector(qi[0:3]).a
        Rj=vector(qj[0:3]).a
        
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        v3=Aj.dot(self.vjk)
        v4=Aj.dot(self.vjj)
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)+10*v3
        
        Hiv1=B(betai,self.vii)
        Hiv2=B(betai,self.vij)
        Hiup=B(betai,self.u_i)
        
        Z=sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[Z,v3.T.dot(Hiv1)],
                         [v1.T,rij.T.dot(Hiv1)+v1.T.dot(Hiup)],
                         [v2.T,rij.T.dot(Hiv2)+v2.T.dot(Hiup)],
                         [Z,v4.T.dot(Hiv1)]],format='csr')
        
        return jac
    
    def jacobian_j(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ai=ep2dcm(betai)
        
        v1=Ai.dot(self.vii)
        v2=Ai.dot(self.vij)
        
        Hjv3=B(betaj,self.vjk)
        Hjv4=B(betaj,self.vjj)
        Hjup=B(betaj,self.u_j)
        
        Z=sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[  Z,   v1.T.dot(Hjv3)],
                         [-v1.T, -v1.T.dot(Hjup)],
                         [-v2.T, -v2.T.dot(Hjup)],
                         [  Z,   v1.T.dot(Hjv4)]],format='csr')
        
        return jac
    
    
    def acc_rhs(self,q,qdot):
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        Ri=qi[0:3].values.reshape((3,1))
        Rj=qj[0:3].values.reshape((3,1))
        betai=qi[3:]
        betaj=qj[3:]
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)

        Ri_dot=qi_dot[0:3].values.reshape((3,1))
        Rj_dot=qj_dot[0:3].values.reshape((3,1))
        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        bid=betai_dot.values.reshape((4,1))
        bjd=betaj_dot.values.reshape((4,1))
        
        v1=self.vii
        v2=self.vij
        v3=self.vjk
        v4=self.vjj
        rij=Ri+Ai.dot(self.u_i)-Rj-Aj.dot(self.u_j)

        
        Biv1=B(betai,v1)
        Biv2=B(betai,v2)
        Bjv3=B(betaj,v3)
        Bjv4=B(betaj,v4)
        Bip=B(betai,self.u_i)
        Bjp=B(betaj,self.u_j)


        Hiv1=B(betai_dot,v1)
        Hiv2=B(betai_dot,v2)
        Hjv3=B(betaj_dot,v3)
        Hjv4=B(betaj_dot,v4)
        Hip =B(betai_dot,self.u_i)
        Hjp =B(betaj_dot,self.u_j)

        
        rij_dot=Ri_dot+Bip.dot(bid)-Rj_dot-Bjp.dot(bjd)
        
        rhs1=acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v3,Aj,Bjv3,Hjv3,bjd)
        rhs3=acc_dp2_rhs(v1,Ai,Biv1,Hiv1,bid,rij,Hip,Hjp,bjd,rij_dot)
        rhs4=acc_dp2_rhs(v2,Ai,Biv2,Hiv2,bid,rij,Hip,Hjp,bjd,rij_dot)
        rhs5=acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v4,Aj,Bjv4,Hjv4,bjd)
        
        return np.concatenate([rhs1,rhs3,rhs4,rhs5])
    

    
    def mir(location,i_body,j_body,axis):
        
        loc_l, loc_r = location
        ibody_l, ibody_r = i_body
        jbody_l, jbody_r = j_body
        
        ax_l, ax_r = axis
        
        left  = cylindrical(loc_l,ibody_l,jbody_l,ax_l)
        right = cylindrical(loc_r,ibody_r,jbody_r,ax_r)
        
        return pd.Series([left,right],index=['l','r'])


class rotational_drive(object):
    def __init__(self,actuated_joint):
        
        self.type='driving constraint'
        self.name=actuated_joint.name+'_actuator'

        self.joint=actuated_joint
        self.v1=actuated_joint.vii
        self.v2=actuated_joint.vji
        self.v3=actuated_joint.vij
        self.i_body=actuated_joint.i_body
        self.j_body=actuated_joint.j_body
        
        self.nc=1
        
        self.pos=0
        self.vel=0
        self.acc=0
        
        self.pos_array=self.vel_array=self.acc_array=[]
    
    
    def set_pos(self,pos,time_array):
        self.pos_array=pos.copy()
        self.vel_array=np.gradient(self.pos_array)/np.gradient(time_array)
        self.acc_array=np.gradient(self.vel_array)/np.gradient(time_array)

    
    def set_vel(self,vel,time_array):
        self.vel_array=vel.copy()
        self.pos_array=sc.integrate.cumtrapz(self.vel_array,time_array,initial=0)
        self.acc_array=np.gradient(self.vel_array)/np.gradient(time_array)

    @property
    def index(self):
        name=self.name
        indices=[name+'_eq%s'%i for i in range(self.nc)]
        return indices
    
    
    def equations(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
                
        Ai=ep2dcm(qi[3:])
        Aj=ep2dcm(qj[3:])
        
        v1=Ai.dot(self.v1)
        v2=Aj.dot(self.v2)
        v3=Ai.dot(self.v3)
        
        c=float(v1.T.dot(v2))
        s=float(v3.T.dot(v2))
        
#        print('angle: %s'%self.pos)
#        
##        print("v1i = %s"%v1.T)
##        print("v2j = %s"%v2.T)
#        print("cos = %s"%c)
#        print("sin = %s"%s)
##        print("%s"%qi[3:])
#        print("%s"%sum(qj[3:]**2))

#        if s>=0 and c>=0:
#            eq=np.arcsin(s)-np.deg2rad(self.pos)
#        if s>=0 and c<0:
#            eq=np.pi-np.arcsin(s)-np.deg2rad(self.pos)
#        if s<0 and c<0:
#            eq=np.pi-np.arcsin(s)-np.deg2rad(self.pos)
#        if s<0 and c>=0:
#            eq=2*np.pi+np.arcsin(s)-np.deg2rad(self.pos)
                
        eq=float(v3.T.dot(v2))-np.sin(np.deg2rad(self.pos))
        
        return np.array([eq])
    
    def jacobian_i(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Aj=ep2dcm(betaj)
        
        v2=Aj.dot(self.v2)
        
        Hiv1 = B(betai,self.v3)
        Z    = sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[Z,v2.T.dot(Hiv1)]],format='csr')
#        print('jaci = %s'%jac.A)
        return jac
    
    def jacobian_j(self,q):
        
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        
        Ai=ep2dcm(betai)
        
        v1=Ai.dot(self.v3)
        
        Hjv2 = B(betaj,self.v2)
        Z    = sparse.csr_matrix([[0,0,0]])
        
        jac=sparse.bmat([[Z,v1.T.dot(Hjv2)]],format='csr')
#        print('jacj = %s'%jac.A)
        return jac
    
    def vel_rhs(self):
        return np.array([[self.vel*np.cos(self.pos)]])
    
    def acc_rhs(self,q,qdot):
        qi=q[self.i_body.dic.index]
        qj=q[self.j_body.dic.index]
        
        qi_dot=qdot[self.i_body.dic.index]
        qj_dot=qdot[self.j_body.dic.index]
        
        betai=qi[3:]
        betaj=qj[3:]
        Ai=ep2dcm(betai)
        Aj=ep2dcm(betaj)

        betai_dot=qi_dot[3:]
        betaj_dot=qj_dot[3:]
        
        bid=betai_dot.values.reshape((4,1))
        bjd=betaj_dot.values.reshape((4,1))
        
        v1=self.v3
        v2=self.v2
        
        Biv1=B(betai,v1)
        Bjv2=B(betaj,v1)


        Hiv1=B(betai_dot,v1)
        Hjv2=B(betaj_dot,v2)
        
        rhs=acc_dp1_rhs(v1,Ai,Biv1,Hiv1,bid,v2,Aj,Bjv2,Hjv2,bjd)+self.acc
        
        return rhs
        
    
class absolute_locating(object):
    def __init__(self,body,coordinate):
        self.type='driving constraint'
        self.name=body.name+'.'+coordinate+'_actuator'
        self.body=body
        self.coordinate=body.name+'.'+coordinate
        self.coo_index=list('xyz').index(coordinate)
        self.nc=1
        
        self.pos=0
        self.vel=0
        self.acc=0
        
        self.pos_array=self.vel_array=self.acc_array=[]
    
    def set_pos(self,pos,time_array):
        self.pos_array=pos.copy()
        self.vel_array=np.gradient(self.pos_array)/np.gradient(time_array)
        self.acc_array=np.gradient(self.vel_array)/np.gradient(time_array)

    
    def set_vel(self,vel,time_array):
        self.vel_array=vel.copy()
        self.pos_array=sc.integrate.cumtrapz(self.vel_array,time_array,initial=0)
        self.acc_array=np.gradient(self.vel_array)/np.gradient(time_array)

    @property
    def index(self):
        name=self.name
        indices=[name+'_eq%s'%i for i in range(self.nc)]
        return indices
    
    
    def equations(self,q):
        ind=q[self.coordinate]
        return np.array([[ind-self.pos]])
    
    def jacobian(self):
        jac=np.zeros((7,))
        jac[self.coo_index]=1
        return jac
    
    def vel_rhs(self):
        return np.array([[self.vel]])
    
    def acc_rhs(self,q=0,qdot=0):
        return np.array([[self.acc]])
    



###############################################################################
###############################################################################
###############################################################################    
    
#class triepod(universal):
#    def __init__(self,location,i_body,j_body,i_rot,j_rot):
#        super().__init__(location,i_body,j_body,i_rot,j_rot)
#        self.type='tripod joint'
#
#    def constraints(self,Ri,Rj,yi,yj):
#        Ri=vector(Ri)
#        Rj=vector(Rj)
#        
#        tm_i=rod2dcm(yi)
#        tm_j=rod2dcm(yj)
#
#        rij=Ri.a+(tm_i.dot(self.u_i))-Rj.a-(tm_j.dot(self.u_j))
#
#        eq1=(tm_i.dot(self.h_i)).T. dot (tm_j.dot(self.h_j))
#        eq2=(tm_i.dot(self.h_j)).T. dot (rij)
#        eq3=(tm_i.dot(self.u_jrf.j)).T. dot (rij)
#        
#        c=[float(i) for i in (eq1,eq2,eq3)]
#        return c
#    
#    def mir(location,i_body,j_body,i_rot,j_rot):
#        
#        loc_l, loc_r = location
#        ibody_l, ibody_r = i_body
#        jbody_l, jbody_r = j_body
#        
#        irot_l, irot_r = i_rot
#        jrot_l, jrot_r = j_rot
#        
#        left  = triepod(loc_l,ibody_l,jbody_l,irot_l,jrot_l)
#        right = triepod(loc_r,ibody_r,jbody_r,irot_r,jrot_r)
#        
#        return pd.Series([left,right],index=['l','r'])
#    
#
#class rack_pinion(joint):
#    def __init__(self,location,i_body,j_body,radius):
#        super().__init__(location,i_body,j_body)
#        self.type='rack_pinion joint'
#        self.raduis=radius
#    
#    def constraints(self,Ri,Rj,yi,yj):
#        Ri=vector(Ri)
#        Rj=vector(Rj)
#        
#        tm_i=rod2dcm(yi)
#        tm_j=rod2dcm(yj)
#        
#        displacement=np.linalg.norm(Rj.a+(tm_j.dot(self.u_j))-self._loc)
#        
#        ji=tm_i.dot(vector([0,1,0]).a)
#        jj=tm_j.dot(vector([0,1,0]).a)
#        
#        eq1=(np.arccos(ji.T.dot(jj))*self.raduis) - displacement
#        
##        print(np.degrees(np.arccos(ji.T.dot(jj))))
##        print((np.arccos(ji.T.dot(jj)))*self.raduis)
##        print(displacement)
#        
#        c=[float(eq1)]
#        return c
#    
#
#class trans(joint):
#    def __init__(self,location,i_body,j_body,axis):
#        super().__init__(location,i_body,j_body,axis)
#        self.type='trans joint'
#        
#    
#    def constraints(self,Ri,Rj,yi,yj):
#        Ri=vector(Ri)
#        Rj=vector(Rj)
#        
#        tm_i=rod2dcm(yi)
#        tm_j=rod2dcm(yj)
#        
#        rij=Ri.a+(tm_i.dot(self.u_i))-Rj.a-(tm_j.dot(self.u_j))
#    
#        eq1=(tm_i.dot(self.vii)).T. dot (tm_j.dot(self.vjk))
#        eq2=(tm_i.dot(self.vij)).T. dot (tm_j.dot(self.vjk))
#        eq3=(tm_i.dot(self.vii)).T. dot (rij)
#        eq4=(tm_i.dot(self.vij)).T. dot (rij)
#        eq5=(tm_i.dot(self.vij)).T. dot (tm_j.dot(self.vji))
#        
#        c=[float(i) for i in (eq1,eq2,eq3,eq4,eq5)]
#        return c

#def acc_dp1_rhs2(joint,q,qdot,v1,v2):
#    qi=q[list(joint.i_body.dic.index)]
#    qj=q[list(joint.j_body.dic.index)]
#    
#    qi_dot=qdot[list(joint.i_body.dic.index)]
#    qj_dot=qdot[list(joint.j_body.dic.index)]
#        
#    betai=qi[3:]
#    betaj=qj[3:]
#    
#    betai_dot=qi_dot[3:]
#    betaj_dot=qj_dot[3:]
#    
#    bid=betai_dot.values.reshape((4,1))
#    bjd=betaj_dot.values.reshape((4,1))
#    
#    Biv1=B(betai,v1)
#    Bjv2=B(betaj,v2)
#    
#    Hiv1=B(betai_dot,v1)
#    Hjv2=B(betaj_dot,v2)
#    
#    Ai=ep2dcm(betai)
#    Aj=ep2dcm(betaj)
#
#    rhs=-2*np.linalg.multi_dot([bid.T,Biv1.T,Bjv2,bjd])-\
#        np.linalg.multi_dot([v2.T,Aj.T,Hiv1,bid])-\
#        np.linalg.multi_dot([v1.T,Ai.T,Hjv2,bjd])
#    
#    return rhs
#
#
#def acc_dp2_rhs2(q,qdot,v1,p):
#    qi=q[list(joint.i_body.dic.index)]
#    qj=q[list(joint.j_body.dic.index)]
#
#    Ri=qi[0:3]
#    Rj=qj[0:3]
#    betai=qi[3:]
#    betaj=qj[3:]
#    Ai=ep2dcm(betai)
#    Aj=ep2dcm(betaj)
#
#    rij=Ri+Ai.dot(p)-Rj-Aj.dot(p)
#    
#
#    qi_dot=qdot[list(joint.i_body.dic.index)]
#    qj_dot=qdot[list(joint.j_body.dic.index)]
#    
#    Ri_dot=qi_dot[0:3]
#    Rj_dot=qj_dot[0:3]
#    betai_dot=qi_dot[3:]
#    betaj_dot=qj_dot[3:]
#    Bip=B(betai_dot,p)
#    Bjp=B(betaj_dot,p)
#    bid=betai_dot.values.reshape((4,1))
#    bjd=betaj_dot.values.reshape((4,1))
#
#    rij_dot=Ri_dot+Bip.dot(bid)-Rj_dot-Bjp.dot(bjd)
#    
#    Biv1= B(betai,v1)
#    Hiv1= B(betai_dot,v1)
#    Hip = B(betai_dot,p)
#    Hjp = B(betaj_dot,p)
#    
#
#    rhs=-2*np.linalg.multi_dot([bid.T,Biv1.T,rij_dot])-\
#        np.linalg.multi_dot([rij.T,Hiv1,bid])-\
#        np.linalg.multi_dot([v1.T,Ai.T,Hip,bid])+\
#        np.linalg.multi_dot([v1.T,Ai.T,Hjp,bjd])
#    
#    return rhs
#
#class sph_sph(joint):
#    def __init__(self,p1,p2,i_body,j_body):
#        super().__init__(p1,i_body,j_body,axis=[0,0,1])
#        
#        self.type='spherical_spherical joint'
#        self.name=self.name+'_sph'
#        self.p2=p2
#        self.d=(p1-p2).mag
#        self.u_j=vector(self.p2-j_body.R).express(j_body).a
#
#
#    
#       
#    def equations(self,q,n):
#        
#        qi=q[n[list(self.i_body.dic.index)]]
#        qj=q[n[list(self.j_body.dic.index)]]
#        
#        Ri=vector(qi[0:3]).a
#        Rj=vector(qj[0:3]).a
#        
#        Ai=ep2dcm(qi[3:])
#        Aj=ep2dcm(qj[3:])
#        
#        p1=self.u_i
#        p2=self.u_j
#        
#        r_p=(Ri+Ai.dot(p1)-Rj-Aj.dot(p2))
#        eq=r_p.T.dot(r_p)-self.d**2
#        return eq
#    
#    def jacobian_i(self,q,n):
#        
#        qi=q[n[list(self.i_body.dic.index)]]
#        qj=q[n[list(self.j_body.dic.index)]]
#        
#        Ri=vector(qi[0:3]).a
#        Rj=vector(qj[0:3]).a
#        
#        Ai=ep2dcm(qi[3:])
#        Aj=ep2dcm(qj[3:])
#        
#        p1=self.u_i
#        p2=self.u_j
#        
#        r_p=(Ri+Ai.dot(p1)-Rj-Aj.dot(p2))
#        
#        Hp=B(qi[3:],p1)
#        
#        jac = np.bmat([[2*r_p.T,2*r_p.T.dot(Hp)]])
#        return jac
#    
#    def jacobian_j(self,q,n):
#        
#        qi=q[n[list(self.i_body.dic.index)]]
#        qj=q[n[list(self.j_body.dic.index)]]
#        
#        Ri=vector(qi[0:3]).a
#        Rj=vector(qj[0:3]).a
#        
#        Ai=ep2dcm(qi[3:])
#        Aj=ep2dcm(qj[3:])
#        
#        p1=self.u_i
#        p2=self.u_j
#        
#        r_p=(Ri+Ai.dot(p1)-Rj-Aj.dot(p2))
#        
#        Hp=B(qj[3:],p2)
#        
#        jac = np.bmat([[-2*r_p.T,-2*r_p.T.dot(Hp)]])
#        return jac
#    

   
