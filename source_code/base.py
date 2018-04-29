# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 05:31:54 2017

@author: khale
"""


# Importing needed directories into the system path
import os
import sys
import numpy as np
import pandas as pd

project_dir=os.getcwd()

f1 = project_dir + '\mbs_objects'
f2 = project_dir + '\solvers'
f3 = project_dir + '\models'
f4 = project_dir + '\jupyter_lab_gui'

path=[f1,f2,f3,f4]
for i in path:
    if i not in sys.path:
        sys.path.append(i)

###############################################################################
# Linear Algebra and Spatial Dynamics common operations.
###############################################################################

def vec2skew(v):
    '''
    converting the vector v into a skew-symetric matrix form
    =======================================================================
    inputs  : 
        v: vector object or ndarray of shape(3,1)
    =======================================================================
    outputs : 
        vs: 3x3 ndarray representing the skew-symmetric form of the vector
    =======================================================================
    '''
    vs=np.array([[0,-v[2,0],v[1,0]],
                 [v[2,0],0,-v[0,0]],
                 [-v[1,0],v[0,0],0]])
    return vs

def skew2vec(vs):
    '''
    converting the skew-symetric matrix  into a vector form
    =======================================================================
    inputs  : 
        vs: 3x3 ndarray representing the skew-symmetric form of the vector
    =======================================================================
    outputs : 
        v: ndarray of shape(3,1)
    ======================================================================= 
    '''
    v1=vs[2,1]
    v2=vs[0,2]
    v3=vs[1,0]
    
    v=np.array([[v1],[v2],[v3]])
    return v

###############################################################################
# These functions are not used in any computations as we adobted euler-
# parameters to define orientation.
###############################################################################
def rot_x(angle):
    mat=np.array([[1,0,0],
                  [0,np.cos(angle),-np.sin(angle)],
                  [0,np.sin(angle), np.cos(angle)]])
    return mat
###############################################################################
def rot_y(angle):
    mat=np.array([[np.cos(angle),0,np.sin(angle)],
                  [0,1,0],
                  [-np.sin(angle),0,np.cos(angle)]])
    return mat
###############################################################################
def rot_z(angle):
    mat=np.array([[np.cos(angle),-np.sin(angle),0],
                  [np.sin(angle),np.cos(angle),0],
                  [0,0,1]])
    return mat   
###############################################################################
def rod2dcm(parameters):
    '''
    Takes in the three rodriguez parameters and returns the corresponding 
    transformation matrix
    '''
    y1,y2,y3=parameters
    y=np.array([[y1],[y2],[y3]])
    mag=np.linalg.norm(y)
    ys=vec2skew(y)
    mat=np.eye(3)+(2/(1+mag**2))*(ys+(ys.dot(ys)))
    
    return mat
###############################################################################
def dcm2rod(mat):
    '''
    Takes in the transformation matrix and returns the corresponding three 
    rodriguez parameters
    '''
    y1=(mat[2,1]-mat[1,2])/(1+mat.trace())
    y2=(mat[0,2]- mat[2,0])/(1 + mat.trace())
    y3=(mat[1,0]- mat[0,1])/(1 + mat.trace())
    
    return y1,y2,y3
    
###############################################################################
# End of unused functions
###############################################################################


###############################################################################
# Linear Algebra and Spatial Dynamics common objects.
###############################################################################

class reference_frame(object):
    def __init__(self,name,orientation=np.eye(3),location=np.zeros((3,1)),parent=False):
        
        # Checking attributes types
        #==========================
        
        name_type  = isinstance(name,str)
        if not name_type : raise TypeError('name should be a string')
        
        loc_type   = isinstance(location,(list,tuple,np.ndarray))
        if not loc_type : raise TypeError('location should be a list, tuple or ndarray')
        loc_length = len(location)==3
        if not loc_length : raise ValueError('location should be a vector of 3 components')
        
        ori_type = isinstance(orientation,np.ndarray)
        if not ori_type : raise TypeError('orientation should be an ndarray')
        ori_shape = orientation.shape==(3,3)
        if not ori_shape : raise ValueError('orientation should be a 3x3 array')
        
        
        if not (isinstance(parent,reference_frame) or not parent) : 
            raise TypeError('parent should be a reference_frame type')
        #======================================================================
        
        self._dcm=orientation
        self.loc=np.reshape(location,(3,1))
        self.T=orientation.T
        
        self.i=self._dcm[:,0].reshape((3,1))
        self.j=self._dcm[:,1].reshape((3,1))
        self.k=self._dcm[:,2].reshape((3,1))
        
        self.name=name
        self.parent=parent
     
    
    def __getitem__(self,key):
        return self.dcm[key]
    
    @property
    def dcm(self):
        return self._dcm
    @dcm.setter
    def dcm(self,value):
        self._dcm=value
        self.i=self._dcm[:,0].reshape((3,1))
        self.j=self._dcm[:,1].reshape((3,1))
        self.k=self._dcm[:,2].reshape((3,1))
        self.T=self._dcm.T
        
    
    
    #==========================================================================
    
    def orient_along_axis(self,name,k_vector,i_vector=None,location=None):
        '''
        Creating a new reference frame along a given vector or 2 orthogonal vectors.
        '''
        
        #Checking the inputs
        k_type   = isinstance(k_vector,(list,tuple,np.ndarray,vector))
        if not k_type : raise TypeError('should be a list, tuple, vector or ndarray')
        k_length = len(k_vector)==3
        if not k_length : raise ValueError('should be a vector of 3 components')

        # Changing the axis type into a vector type 
        try:
            k_vector=k_vector.unit.express(self)
        except AttributeError:
            k_vector=vector(k_vector,self).unit
        
        # generating an arbiatry perpendecular vector to axis1 if only one axis
        # is given
        if i_vector is None:
            dummy=np.array([[1],[1],[1]])
            dummy[np.argmax(abs(k_vector))]=0
            du_v=vector(dummy,self)
            i_vector=k_vector.cross(du_v).unit
        else:
            i_vector=vector(i_vector,self).unit
        
        # Generating Axis 3 and making all axes a unit vectors.
        j_vector=k_vector.cross(i_vector)
        
        # Concatenating the 3 unit orthogonal vector to form a frame matrix
        new_frame=np.concatenate((i_vector,j_vector,k_vector),axis=1)
        
        # Inserting location if given
        loc=(self.loc if location is None else np.array(location).reshape((3,1)))
        # returning the result as a reference_frame object.
        return reference_frame(name,location=loc,orientation=new_frame,parent=self)
     
    ###########################################################################

    def orient_normal(self,name,p1,p2,p3,loc=None):
        '''
        Creating a new reference frame normal to a given 3 points 
        '''
        #Checking the inputs
        for p in (p1,p2,p3):
            p_type   = isinstance(p,(list,tuple,np.ndarray,vector,point))
            if not p_type : raise TypeError('should be a list, tuple, vector or ndarray')
            p_length = len(p)==3
            if not p_length : raise ValueError('should be a vector of 3 components')
        
        
        k_vec=vector.normal(p1,p2,p3,self)
        
        loc=(self.loc if loc is None else np.array(loc).reshape((3,1)))
        
        return self.orient_along_axis(name,k_vec,location=loc)        
    
    ###########################################################################
    
    def orient_rodriguez(self,name,v1,v2,location=None):
        '''
        Reorienting the frame in order to stay in same orientation relative to 
        given vector which changes from v1 to v2 using the Euler-Rodrigues formula
        '''
        v1=vector(v1)
        v2=vector(v2)
        I=np.eye(3)
        V=v1.cross(v2).unit
        V=vec2skew(V)
        theta=np.deg2rad(v1.angle_between(v2))
        R=I+(V*np.sin(theta))+(2*V.dot(V)*np.sin(theta/2)**2)
        R=R.dot(self)
        
        loc=(self.loc if location is None else location)
        return reference_frame(name,orientation=R,location=loc,parent=self)
        
    ###########################################################################

    
    def dot(self,other):
        if isinstance(other,vector): 
            return vector(self.dcm.dot(other.a))
        if isinstance(other,reference_frame): 
            return self.dcm.dot(other.dcm)
        if isinstance(other,np.ndarray): 
            return self.dcm.dot(other)
    
    def __repr__(self):
        return str(self.dcm)
###############################################################################
def orient_along_axis(k_vector,i_vector=None):
    '''
    Creating a new reference frame along a given vector or 2 orthogonal vectors.
    '''
    
    # Changing the axis type into a vector type 
    try:
        k_vector=k_vector.unit
    except AttributeError:
        k_vector=vector(k_vector).unit
    
    # generating an arbiatry perpendecular vector to axis1 if only one axis
    # is given
    if i_vector is None:
        dummy=np.array([[1],[1],[1]])
        dummy[np.argmax(abs(k_vector))]=0
        du_v=vector(dummy)
        i_vector=k_vector.cross(du_v).unit
    else:
        i_vector=vector(i_vector).unit
    
    # Generating Axis 3 and making all axes a unit vectors.
    j_vector=k_vector.cross(i_vector)
    
    # Concatenating the 3 unit orthogonal vector to form a frame matrix
    new_frame=np.concatenate((i_vector,j_vector,k_vector),axis=1)
    
    # returning the result as a reference_frame object.
    return new_frame
###############################################################################
grf=reference_frame('grf')
##############################################################################

class vector(object):
    def __init__(self,components,frame=grf):
        
        # Checking input type and size
#        comp_type   = isinstance(components,(list,tuple,np.ndarray,vector,point,pd.Series))
#        if not comp_type : raise TypeError('should be a list, tuple or ndarray')
#        comp_length = len(components)==3
#        if not comp_length : raise ValueError('should be a vector of 3 components')
        
        self._a=np.array(components).reshape((3,1))    
        self.frame=frame
        self.alignment='S'
    ##########################################################################
    # Defining updatable attributes as a decorated class methods
    # ===========================================================
    
        
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self,value):
        comp_type   = isinstance(value,(list,tuple,np.ndarray,vector,point))
        if not comp_type : raise TypeError('should be a list, tuple or ndarray')
        comp_length = len(value)==3
        if not comp_length : raise ValueError('should be a vector of 3 components')
        self._a=np.array(value).reshape((3,1)) 
    
    @property
    def x(self):
        return self.a[0,0]
    @x.setter
    def x(self, value):
        self.a[0,0] = value
    
    @property
    def y(self):
        return self.a[1,0]
    @y.setter
    def y(self, value):
        self.a[1,0] = value
    
    @property
    def z(self):
        return self.a[2,0]
    @z.setter
    def z(self, value):
        self.a[2,0] = value
        
    @property
    def p1(self):
        return self._p1
    @p1.setter
    def p1(self,value):
        self._p1=value
    
    @property
    def p2(self):
        return self._p2
    @p2.setter
    def p2(self,value):
        self._p2=value
    
    @property
    def p3(self):
        return self._p3
    @p3.setter
    def p3(self,value):
        self._p3=value
        
    
    @property
    def mag(self):
        return np.linalg.norm(self.a)
    
    @property
    def T(self):
        return self.a.T
    
    @property
    def row(self):
        return np.resize(self.a,(3,))
    
    @property
    def unit(self):
        return vector(self.a/self.mag,self.frame)

    @property
    def cosines(self):
        cosiens=self.unit
        alpha =np.degrees(np.arccos(cosiens.x))
        beta  =np.degrees(np.arccos(cosiens.y))
        gamma =np.degrees(np.arccos(cosiens.z))
        return [alpha,beta,gamma]
    

    # #########################################################################
    # Defining built-in class methods
    # ===========================================================
    
    def _framecheck(self,other):
        if self.frame.name != other.frame.name: 
            raise ValueError('the given vectors are in different frames')
        
    def __add__(self,other):
        self._framecheck(other)
        return vector(self.a + other.a,self.frame)
    
    def __sub__(self,other):
        self._framecheck(other)
        return vector(self.a - other.a,self.frame)
    
    def __neg__(self):
        return vector(-self.a,self.frame)
    
    def __mul__(self,other):
        return vector(other*self.a,self.frame)
    
    def __rmul__(self,other):
        return vector(other*self.a,self.frame)
    
    def __truediv__(self,other):
        return vector(self.a/other,self.frame)
    
    def __rtruediv__(self,other):
        return vector(self.a/other,self.frame)
    
    def __getitem__(self,key):
        return self.a[key]
    
    def __len__(self):
        return len(self.a)
    
    def __abs__(self):
        return vector(abs(self.a),self.frame)
    
    @property
    def copy(self):
        return vector(self.a,self.frame)
    
    def dot(self,other):
        self._framecheck(other)
        return float(np.dot(self.T,other.a))
    
    def cross(self,other):
        self._framecheck(other)
        v=vec2skew(self.a).dot(other.a)
        return vector(v,frame=self.frame)

    def angle_between(self,other):
        self._framecheck(other)
        return float(np.rad2deg(np.arccos(self.dot(other)/(self.mag*other.mag))))
    
    
    def _normal(name,p1,p2,p3,frame=grf):
        '''
        return a vector normal to a plane defined by 3 points.
        To be used as vector.normal(args)
        '''
        v1=p1-p2
        v2=p1-p3
        normal_vec=point(name,v2.cross(v1).unit)
        
        normal_vec._p1=p1
        normal_vec._p2=p2
        normal_vec._p3=p3
        
        return normal_vec
    
    @property
    def normal(self,name,p1,p2,p3):
        return self._normal
    
    
    @normal.setter
    def normal(self,p):
        self.p1=(p if p.name==self.p1.name else self.p1)
        self.p2=(p if p.name==self.p2.name else self.p2)
        self.p3=(p if p.name==self.p3.name else self.p3)
        v1=self.p1-self.p2
        v2=self.p1-self.p3
        normal_vec=point(self.name,v2.cross(v1).unit)
        self.x,self.y,self.z = normal_vec.x,normal_vec.y,normal_vec.z
        
   
    def _a2b(name,p1,p2):
        v = point(name,p2-p1)
        v._p1=p1
        v._p2=p2
        return v
    
    @property
    def a2b(self):
        return self._a2b(self.name,self.p1,self.p2)
    @a2b.setter
    def a2b(self,p):
        self.p1=(p if p.name==self.p1.name else self.p1)
        self.p2=(p if p.name==self.p2.name else self.p2)
        v = point(self.name,self.p2-self.p1)
        self.x,self.y,self.z = v.x,v.y,v.z
        
    
    def express(self,other_frame):
        '''
        Expressing a vector in terms of other frame base vectors using the
        recursive transformation function
        '''
        tm=recursive_transformation(self.frame,other_frame)
        new=tm.dot(self.a)
        return vector(new,other_frame)
    

    
    def __repr__(self):
        return str(self.a)
###############################################################################

###############################################################################
class point(vector):
    '''
    A point class representing the poinnt in space and stores the required data
    for further use in the MBS analysis
    '''
    
    def __init__(self,name,components,frame=grf,body=None):
        super().__init__(components,frame)
        
        # Checking input type and size
        comp_type   = isinstance(components,(list,tuple,np.ndarray,vector,point,pd.Series))
        if not comp_type : raise TypeError('should be a list, tuple or ndarray')
        comp_length = len(components)==3
        if not comp_length : raise ValueError('should be a vector of 3 components')
        
        
        self.name=name
        self.alignment='S'
        self._body=body
        self.u_i=(None if body==None else vector(self-body.loc).express(body))
        self.notes=''
    
    @property    
    def m_name(self):
        if self.alignment=='S':
            return 'hps_'+self.name[4:]
        elif self.alignment == 'R':
            return 'hpl_'+self.name[4:]
        elif self.alignment == 'L':
            return 'hpr_'+self.name[4:]
    
    @property    
    def m_object(self):
        y=-self.y
        alignment = 'RL'.replace(self.alignment,'')
        loc=[self.x,y,self.z]
        name=self.m_name
        p = point(name,loc)
        p.alignment = alignment
        return p
    
    @property
    def right(self):
        return point(self.name+'.r',self._right,typ='r')
    @property
    def left(self):
        return point(self.name+'.r',self._left,typ='l')
        
    @property
    def body(self):
        return self._body
    @body.setter
    def body(self,value):
        self._body=value
        body=self._body
        self.u_i=(self-body.R).express(body)
    
    @property
    def dic(self):
        name=self.name+'.'
        return {name+'x':self.x,name+'y':self.y,name+'z':self.z}
    
    def _mid_point(name,p1,p2):
                
        loc=((p2-p1).mag/2)*(p2-p1).unit
        loc=loc+p1
        mid=point(name,loc)
        mid._p1=p1
        mid._p2=p2
        
        return mid
    
    @property
    def mid_point(self):
        return self._mid_point(self.name,self.p1,self.p2)
    @mid_point.setter
    def mid_point(self,p):
        self.p1=(p if p.name==self.p1.name else self.p1)
        self.p2=(p if p.name==self.p2.name else self.p2)
        v = point(self.name,self.p2-self.p1)
        self.x,self.y,self.z = v.x,v.y,v.z
    
    def point_pos(self,R,rod_param):
        R=vector(R)
        tm=rod2dcm(rod_param)
        pos=R.a+tm.dot(self.u_i)
        return vector(pos)
    
        
    @property
    def mir_sys(self):
        r_loc    = self.copy
        l_loc    = self.copy
        r_loc.y  = abs(r_loc.y)
        l_loc.y  = -r_loc.y
        
        right = point(self.name+'.r',r_loc,typ='r')
        left  = point(self.name+'.l',l_loc,typ='l')
        
        return pd.Series([left,right],index=['l','r'])
    
    def mir(name,location,model=None):
        
        if location[1] <0:
            l_coord=location
            r_coord=location.copy(); r_coord[1]=-1*r_coord[1]
        elif location[1]>0:
            r_coord=location
            l_coord=location.copy(); l_coord[1]=-1*r_coord[1]
        elif location[1]==0:
            s_coord=location
            single  = point(name+'.s',s_coord,typ='s')
            return pd.Series([single],index=['l'])
            
        left  = point(name+'.l',l_coord,typ='l')
        right = point(name+'.r',r_coord,typ='r')
        p     = pd.Series([left,right],index=['l','r'])
        
        
        return p
        
    
        
     
    def __repr__(self):
        return "point object at " +str(self.row)
###############################################################################
###############################################################################
def recursive_transformation(frame_n,frame_m,global_ref=grf):
    '''
    This function produces the transformation matrix from frame n to frame m by
    refering frame n to its parents untill finding frame m. If frame m is not a 
    parent, then refering both to global frame then refer the global to m so that
    refering n--> global & global--> m therefore n_m = [To]m*[Tn]o
    '''
    
    if np.all(frame_n.dcm==frame_m.dcm):
        return np.eye(3)
    
    elif frame_n.name=='grf':
        return recursive_transformation(frame_m,frame_n).T
    
    elif frame_n.parent.name==frame_m.name:
        return frame_n
    
    else:
        if frame_n.parent.name=='grf':
            print('reached global')
            return recursive_transformation(frame_m,global_ref).T.dot(
                          recursive_transformation(frame_n,global_ref))
        else:
            print('recursing')
            return recursive_transformation(frame_n.parent,frame_m).dot(frame_n.mat)

##############################################################################


###############################################################################
# The mostly used functions are below. These define most of the matrices derived
# from euler-parameters unit quaternion.
###############################################################################

def rot2ep(theta,v):
    '''
    evaluating euler parameters from euler-rodriguez theorem.
    ===========================================================================
    inputs  : 
        theta : angle of rotation of the reference frame in degrees
        v     : axis of rotation as a numerical object of length 3
    ===========================================================================
    outputs : 
        p   : tuple containing the four parameters
    ===========================================================================
    '''
    u=vector(v).unit
    e0=np.cos(np.deg2rad(theta)/2)
    e1=u.x*np.sin(np.deg2rad(theta)/2)
    e2=u.y*np.sin(np.deg2rad(theta)/2)
    e3=u.z*np.sin(np.deg2rad(theta)/2)
    
    return e0,e1,e2,e3


def ep2dcm2(p):
    '''
    evaluating the transformation matrix as a function of euler parameters
    Note: The matrix is defined explicitly by defining the its elements.
    ===========================================================================
    inputs  : 
        p   : set "any list-like object" containing the four parameters
    ===========================================================================
    outputs : 
        A   : The transofrmation matrix 
    ===========================================================================
    '''
    e0,e1,e2,e3=p
    A=2*np.array([[0.5*(e0**2+e1**2-e2**2-e3**2), (e1*e2)-(e0*e3)               , (e1*e3)+(e0*e2)], 
                  [(e1*e2)+(e0*e3)              , 0.5*(e0**2-e1**2+e2**2-e3**2) , (e2*e3)-(e0*e1)], 
                  [(e1*e3)-(e0*e2)              , (e2*e3)+(e0*e1)               , 0.5*(e0**2-e1**2-e2**2+e3**2)]])
    return A

def ep2dcm(p):
    '''
    evaluating the transformation matrix as a function of euler parameters
    Note: The matrix is defined as a prodcut of the two special matrices 
    of euler parameters, the E and G matrices. This function is faster.
    ===========================================================================
    inputs  : 
        p   : set "any list-like object" containing the four parameters
    ===========================================================================
    outputs : 
        A   : The transofrmation matrix 
    ===========================================================================
    '''
    return E(p).dot(G(p).T)
    
def dcm2ep(dcm):
    ''' 
    extracting euler parameters from a transformation matrix
    Note: this is not fully developed. The special case of e0=0 is not dealt 
    with yet.
    ===========================================================================
    inputs  : 
        A   : The transofrmation matrix
    ===========================================================================
    outputs : 
        p   : set containing the four parameters
    ===========================================================================
    '''
    e0=np.sqrt(1-(dcm.trace()-3)/-4)
    
    # Case 1 : e0 != zero
    if abs(e0)>=1e-15:
        e1=(dcm[2,1]-dcm[1,2])/(4*e0)
        e2=(dcm[0,2]-dcm[2,0])/(4*e0)
        e3=(dcm[1,0]-dcm[0,1])/(4*e0)
        
        if abs(dcm[1,0]-(2*(e1*e2+e0*e3)))>=0.0000001:
            e0=-1*e0
            e1=(dcm[2,1]-dcm[1,2])/(4*e0)
            e2=(dcm[0,2]-dcm[2,0])/(4*e0)
            e3=(dcm[1,0]-dcm[0,1])/(4*e0)
    else: # for now
        raise ValueError('e0 is zero for the given orientation matrix')
    
    return e0,e1,e2,e3

def E(p):
    ''' 
    A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the global coordinate system.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
    ===========================================================================
    outputs : 
        E   : 3x4 ndarray
    ===========================================================================
    '''
    e0,e1,e2,e3=p
    m=np.array([[-e1, e0,-e3, e2],
                [-e2, e3, e0,-e1],
                [-e3,-e2, e1, e0]])
    return m

def G(p):
    # Note: This is half the G_bar given in shabana's book
    ''' 
    A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the body coordinate system.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
    ===========================================================================
    outputs : 
        G   : 3x4 ndarray
    ===========================================================================
    '''
    e0,e1,e2,e3=p
    m=np.array([[-e1, e0, e3,-e2],
                [-e2,-e3, e0, e1],
                [-e3, e2,-e1, e0]])
    return m


def B(p,a):
    ''' 
    This matrix represents the variation of the body orientation with respect
    to the change in euler parameters. This can be thought as the jacobian of
    the A.dot(a), where A is the transformation matrix in terms of euler 
    parameters.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
        a   : vector 
    ===========================================================================
    outputs : 
        B   : 3x4 ndarray representing the jacobian of A.dot(a)
    ===========================================================================
    '''
    e0,e1,e2,e3=p
    e=np.array([[e1],[e2],[e3]])
    a=np.array(a).reshape((3,1))
    a_s=vec2skew(a)
    I=np.eye(3)
    e_s=vec2skew(e)
    
    m=2*np.bmat([[(e0*I+e_s).dot(a), e.dot(a.T)-(e0*I+e_s).dot(a_s)]])
    
    return m

def B_exp(p,u):
    ''' 
    Same as the B(p,a) function, except this is defined explicitly using hand
    calculations and checked by sympy symbolic jacobian function
    IMPORTANT NOTE: The jacobian depeneds heavily on how matrix A is defined.
    As the diagonal elements of A can be defined using several formulations 
    using the normalization property of euler parameters :
        e0**2+e1**2+e2**2+e3**2=1
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
        a   : vector 
    ===========================================================================
    outputs : 
        B   : 3x4 ndarray representing the jacobian of A.dot(a)
    ===========================================================================
    '''
    e0,e1,e2,e3=p
    ux,uy,uz=vector(u).row
    
    m=np.array([[2*e0*ux + 2*e2*uz - 2*e3*uy, 2*e1*ux + 2*e2*uy + 2*e3*uz, 2*e0*uz + 2*e1*uy - 2*e2*ux, -2*e0*uy + 2*e1*uz - 2*e3*ux],
                [2*e0*uy - 2*e1*uz + 2*e3*ux, -2*e0*uz - 2*e1*uy + 2*e2*ux, 2*e1*ux + 2*e2*uy + 2*e3*uz, 2*e0*ux + 2*e2*uz - 2*e3*uy],
                [2*e0*uz + 2*e1*uy - 2*e2*ux, 2*e0*uy - 2*e1*uz + 2*e3*ux, -2*e0*ux - 2*e2*uz + 2*e3*uy, 2*e1*ux + 2*e2*uy + 2*e3*uz]])

    return m

