# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:32:48 2018

@author: Khaled Ghobashy
"""




from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from inertia_properties import composite_geometry, triangular_prism
from constraints import spherical, revolute, universal, \
cylindrical, rotational_drive, absolute_locating,translational,bounce_roll
from force_elements import force, tire_force,air_strut
from pre_processor import topology_writer
import pandas as pd
import numpy as np
from solvers import kds, check_jacobian_dense, reactions, dds, state_space_creator
from newton_raphson import nr_kds
import matplotlib.pyplot as plt
import simulations_subroutines as ss


###############################################################################
# Defining System HardPoints.
###############################################################################
# Right Rear.
###############################################################################

ucaf_rr   = point('ucaf_rr',   [-3673 ,334, 807])
ucar_rr   = point('ucar_rr',   [-3933 ,334, 807])
ucao_rr   = point('ucao_rr',   [-3803 ,812, 865])

lcaf_rr   = point('lcaf_rr',   [-3463 ,269, 527])
lcar_rr   = point('lcar_rr',   [-4143 ,269, 527])
lcao_rr   = point('lcao_rr',   [-3803 ,841, 453])

ch_sh_rr  = point('ch_sh_rr',  [-3982 ,588, 1251])
sh_lca_rr = point('sh_lca_rr', [-3971 ,588, 539])

tro_rr    = point('tro_rr',    [-4217 ,800, 604])
tri_rr    = point('tri_rr',    [-4217 ,278, 631]) #assumed

wc_rr     = point('wc_rr',     [-3803 ,1100, 600])
cp_rr     = point('cp_rr',     [-3803 ,1100, 0.0])

d_m_rr    = point.mid_point(ch_sh_rr,sh_lca_rr,'d_m_rr')

###############################################################################
# Left Rear.
###############################################################################


ucaf_lr   = point('ucaf_lr',   [-3673 ,-334, 807])
ucar_lr   = point('ucar_lr',   [-3933 ,-334, 807])
ucao_lr   = point('ucao_lr',   [-3803 ,-812, 865])

lcaf_lr   = point('lcaf_lr',   [-3463 ,-269, 527])
lcar_lr   = point('lcar_lr',   [-4143 ,-269, 527])
lcao_lr   = point('lcao_lr',   [-3803 ,-841, 453])

ch_sh_lr  = point('ch_sh_lr',  [-3982 ,-588, 1251])
sh_lca_lr = point('sh_lca_lr', [-3971 ,-588, 539])

tro_lr    = point('tro_lr',    [-4217 ,-800, 604])
tri_lr    = point('tri_lr',    [-4217 ,-278, 631]) #assumed

wc_lr     = point('wc_lr',     [-3803 ,-1100, 600])
cp_lr     = point('cp_lr',     [-3803 ,-1100, 0.0])

d_m_lr    = point.mid_point(ch_sh_lr,sh_lca_lr,'d_m_lr')

###############################################################################
# Right Front.
###############################################################################
ch_sh_rf  = point('ch_sh_rf',  [-3982+3803 ,628, 1251])
sh_lca_rf = point('sh_lca_rf', [-3971+3803 ,586, 508])

tro_rf    = point('tro_rf',    [-4217+3803 ,788, 669])
tri_rf    = point('tri_rf',    [-4217+3803 ,285, 669]) #assumed

ucaf_rf   = point('ucaf_rf',   [-3673+3803 ,334, 807])
ucao_rf   = point('ucao_rf',   [-3803+3803 ,812, 865])
ucar_rf   = point('ucar_rf',   [-3933+3803 ,334, 807])

lcaf_rf   = point('lcaf_rf',   [-3463+3803 ,269, 527])
lcao_rf   = point('lcao_rf',   [-3803+3803 ,848, 453])
lcar_rf   = point('lcar_rf',   [-4143+3803 ,269, 527])

wc_rf     = point('wc_rf',     [-3803+3803  ,1100, 600])
cp_rf     = point('cp_rf',     [-3803+3803  ,1100, 0.0])

d_m_rf    = point.mid_point(ch_sh_rf,sh_lca_rf,'d_m_rf')

###############################################################################
# Left Front.
###############################################################################
ch_sh_lf  = point('ch_sh_lf',  [-3982+3803 ,-628, 1251])
sh_lca_lf = point('sh_lca_lf', [-3971+3803 ,-586, 508])

tro_lf    = point('tro_lf',    [-4217+3803 ,-788, 669])
tri_lf    = point('tri_lf',    [-4217+3803 ,-285, 669]) #assumed

ucaf_lf   = point('ucaf_lf',   [-3673+3803 ,-334, 807])
ucao_lf   = point('ucao_lf',   [-3803+3803 ,-812, 865])
ucar_lf   = point('ucar_lf',   [-3933+3803 ,-334, 807])

lcaf_lf   = point('lcaf_lf',   [-3463+3803 ,-269, 527])
lcao_lf   = point('lcao_lf',   [-3803+3803 ,-848, 453])
lcar_lf   = point('lcar_lf',   [-4143+3803 ,-269, 527])

wc_lf     = point('wc_lf',     [-3803+3803  ,-1100, 600])
cp_lf     = point('cp_lf',     [-3803+3803  ,-1100, 0.0])

d_m_lf    = point.mid_point(ch_sh_lf,sh_lca_lf,'d_m_lf')


###############################################################################
# Defining System Bodies and their inertia properties.
###############################################################################
I=np.eye(3)
cm=vector([0,0,0])
dcm=I
J=I
mass=1
ground  = rigid('ground',mass,J,cm,dcm,typ='mount')
########################################################################
#Chassis
########
ch_cm=vector([-3803/2,0,1800])
ch_dcm=I
ch_J=I
ch_mass  = 16000*1e3
chassis  = rigid('chassis',ch_mass,ch_J,ch_cm,ch_dcm)
###############################################################################


###############################################################################
# Right Rear.
###############################################################################
###############################################################################
tube1    = circular_cylinder(ucaf_rr,ucao_rr,40,0)
tube2    = circular_cylinder(ucar_rr,ucao_rr,40,0)
uca_g    = composite_geometry([tube1,tube2])
uca_rr   = rigid('uca_rr',uca_g.mass,uca_g.J,uca_g.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf_rr,lcao_rr,40,0)
tube2    = circular_cylinder(lcar_rr,lcao_rr,40,0)
lca_g    = composite_geometry([tube1,tube2])
lca_rr   = rigid('lca_rr',lca_g.mass,lca_g.J,lca_g.cm,I)
########################################################################

upright_tube = circular_cylinder(lcao_rr,ucao_rr,50,0)
hub_cylinder = circular_cylinder(point.mid_point(lcao_rr,ucao_rr,"up_rr"),wc_rr,400,350)
upright_geo  = composite_geometry([upright_tube,hub_cylinder])

upright_rr   = rigid('upright_rr',upright_geo.mass,upright_geo.J,upright_geo.cm,I)
########################################################################
tie_g  = circular_cylinder(tri_rr,tro_rr,40,0)
tie_rr = rigid('tie_rr',tie_g.mass,tie_g.J,tie_g.cm,tie_g.C)
########################################################################
d1_g  = circular_cylinder(sh_lca_rr,d_m_rr,40)
cm    = d1_g.cm
dcm   = d1_g.C
J     = d1_g.J
mass  = d1_g.mass 
d1_rr = rigid('d1_rr',mass,J,cm,dcm)
########################################################################
d2_g  = circular_cylinder(ch_sh_rr,d_m_rr,60,28)
cm    = d2_g.cm
dcm   = d2_g.C
J     = d2_g.J
mass  = d2_g.mass 
d2_rr = rigid('d2_rr',mass,J,cm,dcm)
########################################################################
cm     = vector([-3803,1100.5,600])
Jcm=np.array([[343952295.71, 29954.40     , -40790.37    ],
              [29954.40    , 535366217.28 , -28626.24    ],
              [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 80*1e3  
wheel_rr  = rigid('wheel_rr',mass,J,cm,I)
###############################################################################


###############################################################################
# Left Rear.
###############################################################################
###############################################################################
tube1    = circular_cylinder(ucaf_lr,ucao_lr,40,0)
tube2    = circular_cylinder(ucar_lr,ucao_lr,40,0)
uca_g    = composite_geometry([tube1,tube2])
uca_lr   = rigid('uca_lr',uca_g.mass,uca_g.J,uca_g.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf_lr,lcao_lr,40,0)
tube2    = circular_cylinder(lcar_lr,lcao_lr,40,0)
lca_g    = composite_geometry([tube1,tube2])
lca_lr   = rigid('lca_lr',lca_g.mass,lca_g.J,lca_g.cm,I)
########################################################################

upright_tube = circular_cylinder(lcao_lr,ucao_lr,50,0)
hub_cylinder = circular_cylinder(point.mid_point(lcao_lr,ucao_lr,"up_lr"),wc_lr,400,350)
upright_geo  = composite_geometry([upright_tube,hub_cylinder])

upright_lr   = rigid('upright_lr',upright_geo.mass,upright_geo.J,upright_geo.cm,I)
########################################################################
tie_g   = circular_cylinder(tri_lr,tro_lr,40,0)
tie_lr  = rigid('tie_lr',tie_g.mass,tie_g.J,tie_g.cm,tie_g.C)
########################################################################
d1_g  = circular_cylinder(sh_lca_lr,d_m_lr,40)
cm    = d1_g.cm
dcm   = d1_g.C
J     = d1_g.J
mass  = d1_g.mass 
d1_lr = rigid('d1_lr',mass,J,cm,dcm)
########################################################################
d2_g  = circular_cylinder(ch_sh_lr,d_m_lr,60,28)
cm    = d2_g.cm
dcm   = d2_g.C
J     = d2_g.J
mass  = d2_g.mass 
d2_lr = rigid('d2_lr',mass,J,cm,dcm)
########################################################################
cm     = vector([-3803,-1100.5,600])
Jcm=np.array([[343952295.71, 29954.40     , -40790.37    ],
              [29954.40    , 535366217.28 , -28626.24    ],
              [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 80*1e3  
wheel_lr  = rigid('wheel_lr',mass,J,cm,I)
###############################################################################


###############################################################################
# Right Front.
###############################################################################
###############################################################################
tube1    = circular_cylinder(ucaf_rf,ucao_rf,40,0)
tube2    = circular_cylinder(ucar_rf,ucao_rf,40,0)
uca_g    = composite_geometry([tube1,tube2])
uca_rf   = rigid('uca_rf',uca_g.mass,uca_g.J,uca_g.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf_lr,lcao_lr,40,0)
tube2    = circular_cylinder(lcar_lr,lcao_lr,40,0)
lca_g    = composite_geometry([tube1,tube2])
lca_rf   = rigid('lca_rf',lca_g.mass,lca_g.J,lca_g.cm,I)
########################################################################

upright_tube = circular_cylinder(lcao_rf,ucao_rf,50,0)
hub_cylinder = circular_cylinder(point.mid_point(lcao_rf,ucao_rf,"up_rf"),wc_rf,400,350)
upright_geo  = composite_geometry([upright_tube,hub_cylinder])

upright_rf   = rigid('upright_rf',upright_geo.mass,upright_geo.J,upright_geo.cm,I)
########################################################################
tie_g   = circular_cylinder(tri_rf,tro_rf,40,0)
tie_rf  = rigid('tie_rf',tie_g.mass,tie_g.J,tie_g.cm,tie_g.C)
########################################################################
d1_g  = circular_cylinder(sh_lca_rf,d_m_rf,40)
cm    = d1_g.cm
dcm   = d1_g.C
J     = d1_g.J
mass  = d1_g.mass 
d1_rf = rigid('d1_rf',mass,J,cm,dcm)
########################################################################
d2_g  = circular_cylinder(ch_sh_rf,d_m_rf,60,28)
cm    = d2_g.cm
dcm   = d2_g.C
J     = d2_g.J
mass  = d2_g.mass 
d2_rf = rigid('d2_rf',mass,J,cm,dcm)
########################################################################
cm     = vector([-3803+3803,1100.5,600])
Jcm    =np.array([[343952295.71, 29954.40     , -40790.37    ],
                  [29954.40    , 535366217.28 , -28626.24    ],
                  [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 80*1e3  
wheel_rf  = rigid('wheel_rf',mass,J,cm,I)
###############################################################################


###############################################################################
# Left Front.
###############################################################################
###############################################################################
tube1    = circular_cylinder(ucaf_lf,ucao_lf,40,0)
tube2    = circular_cylinder(ucar_lf,ucao_lf,40,0)
uca_g    = composite_geometry([tube1,tube2])
uca_lf   = rigid('uca_lf',uca_g.mass,uca_g.J,uca_g.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf_lf,lcao_lf,40,0)
tube2    = circular_cylinder(lcar_lf,lcao_lf,40,0)
lca_g    = composite_geometry([tube1,tube2])
lca_lf   = rigid('lca_lf',lca_g.mass,lca_g.J,lca_g.cm,I)
########################################################################

upright_tube = circular_cylinder(lcao_lf,ucao_lf,50,0)
hub_cylinder = circular_cylinder(point.mid_point(lcao_lf,ucao_lf,"up_lf"),wc_lf,400,350)
upright_geo  = composite_geometry([upright_tube,hub_cylinder])

upright_lf   = rigid('upright_lf',upright_geo.mass,upright_geo.J,upright_geo.cm,I)
########################################################################
tie_g   = circular_cylinder(tri_lf,tro_lf,40,0)
tie_lf  = rigid('tie_lf',tie_g.mass,tie_g.J,tie_g.cm,tie_g.C)
########################################################################
d1_g  = circular_cylinder(sh_lca_lf,d_m_lf,40)
cm    = d1_g.cm
dcm   = d1_g.C
J     = d1_g.J
mass  = d1_g.mass 
d1_lf = rigid('d1_lf',mass,J,cm,dcm)
########################################################################
d2_g  = circular_cylinder(ch_sh_lf,d_m_lf,60,28)
cm    = d2_g.cm
dcm   = d2_g.C
J     = d2_g.J
mass  = d2_g.mass 
d2_lf = rigid('d2_lf',mass,J,cm,dcm)
########################################################################
cm     = vector([-3803+3803,-1100.5,600])
Jcm    = np.array([[343952295.71, 29954.40     , -40790.37    ],
                   [29954.40    , 535366217.28 , -28626.24    ],
                   [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 80*1e3  
wheel_lf  = rigid('wheel_lf',mass,J,cm,I)
###############################################################################


###############################################################################
# Defining System Joints.
###############################################################################

###############################################################################
# Right Rear Joints.
###############################################################################

uca_rev_rr     = revolute(ucaf_rr,uca_rr,chassis,ucaf_rr-ucar_rr)
lca_rev_rr     = revolute(lcaf_rr,lca_rr,chassis,lcaf_rr-lcar_rr)
wheel_hub_rr   = revolute(wc_rr,wheel_rr,upright_rr,vector([0,-1,0]))

tie_up_sph_rr  = spherical(tro_rr,tie_rr,upright_rr)
ucao_sph_rr    = spherical(ucao_rr,uca_rr,upright_rr)
lcao_sph_rr    = spherical(lcao_rr,lca_rr,upright_rr)

damper_rr      = cylindrical(d_m_rr,d1_rr,d2_rr,sh_lca_rr-ch_sh_rr)

d1_uni_rr      = universal(sh_lca_rr,lca_rr, d1_rr     ,sh_lca_rr-d_m_rr,sh_lca_rr-d_m_rr)
d2_uni_rr      = universal(ch_sh_rr ,d2_rr , chassis,sh_lca_rr-d_m_rr,sh_lca_rr-d_m_rr)

wheel_drive_rr = rotational_drive(wheel_hub_rr)
###############################################################################



###############################################################################
# Left Rear Joints.
###############################################################################

uca_rev_lr     = revolute(ucaf_lr,uca_lr,chassis,ucaf_lr-ucar_lr)
lca_rev_lr     = revolute(lcaf_lr,lca_lr,chassis,lcaf_lr-lcar_lr)
wheel_hub_lr   = revolute(wc_lr,wheel_lr,upright_lr,vector([0,-1,0]))

tie_up_sph_lr  = spherical(tro_lr,tie_lr,upright_lr)
ucao_sph_lr    = spherical(ucao_lr,uca_lr,upright_lr)
lcao_sph_lr    = spherical(lcao_lr,lca_lr,upright_lr)

damper_lr      = cylindrical(d_m_lr,d1_lr,d2_lr,sh_lca_lr-ch_sh_lr)

d1_uni_lr      = universal(sh_lca_lr,lca_lr, d1_lr     ,sh_lca_lr-d_m_lr,sh_lca_lr-d_m_lr)
d2_uni_lr      = universal(ch_sh_lr ,d2_lr , chassis,sh_lca_lr-d_m_lr,sh_lca_lr-d_m_lr)

wheel_drive_lr = rotational_drive(wheel_hub_lr)
###############################################################################



###############################################################################
# Right Front Joints.
###############################################################################

uca_rev_rf     = revolute(ucaf_rf,uca_rf,chassis,ucaf_rf-ucar_rf)
lca_rev_rf     = revolute(lcaf_rf,lca_rf,chassis,lcaf_rf-lcar_rf)
wheel_hub_rf   = revolute(wc_rf,wheel_rf,upright_rf,vector([0,-1,0]))

tie_up_sph_rf  = spherical(tro_rf,tie_rf,upright_rf)
ucao_sph_rf    = spherical(ucao_rf,uca_rf,upright_rf)
lcao_sph_rf    = spherical(lcao_rf,lca_rf,upright_rf)

damper_rf      = cylindrical(d_m_rf,d1_rf,d2_rf,sh_lca_rf-ch_sh_rf)

d1_uni_rf      = universal(sh_lca_rf,lca_rf, d1_rf     ,sh_lca_rf-d_m_rf,sh_lca_rf-d_m_rf)
d2_uni_rf      = universal(ch_sh_rf ,d2_rf , chassis,sh_lca_rf-d_m_rf,sh_lca_rf-d_m_rf)

wheel_drive_rf = rotational_drive(wheel_hub_rf)
###############################################################################

###############################################################################
# Left Front Joints.
###############################################################################

uca_rev_lf     = revolute(ucaf_lf,uca_lf,chassis,ucaf_lf-ucar_lf)
lca_rev_lf     = revolute(lcaf_lf,lca_lf,chassis,lcaf_lf-lcar_lf)
wheel_hub_lf   = revolute(wc_lf,wheel_lf,upright_lf,vector([0,-1,0]))

tie_up_sph_lf  = spherical(tro_lf,tie_lf,upright_lf)
ucao_sph_lf    = spherical(ucao_lf,uca_lf,upright_lf)
lcao_sph_lf    = spherical(lcao_lf,lca_lf,upright_lf)

damper_lf      = cylindrical(d_m_lf,d1_lf,d2_lf,sh_lca_lf-ch_sh_lf)

d1_uni_lf      = universal(sh_lca_lf,lca_lf, d1_lf     ,sh_lca_lf-d_m_lf,sh_lca_lf-d_m_lf)
d2_uni_lf      = universal(ch_sh_lf ,d2_lf , chassis,sh_lca_lf-d_m_lf,sh_lca_lf-d_m_lf)

wheel_drive_lf = rotational_drive(wheel_hub_lf)
###############################################################################

###############################################################################
# Front STEARING MECHANISM
###############################################################################

mount_1_front   = point("mount_1_front" , [-4500+3803  , 286  ,54+19+600])
mount_2_front   = point("mount_2_front" , [-4500+3803  ,-286  ,54+19+600])
coupler_1_front = point("C1_front"      , [-4376+3803  , 364  ,50+19+600])
coupler_2_front = point("C2_front"      , [-4376+3803  ,-364  ,50+19+600])
E_front         = point("E_front"       , [-4320+3803  , 608  ,157+19+600])
F_front         = point("F_front"       , [-4349+3803  ,-285  ,85+19+600])
EF_front        = point.mid_point(E_front,F_front,'EF_front')


l1g = circular_cylinder(mount_1_front,coupler_1_front,40)
l2g = circular_cylinder(coupler_1_front,coupler_2_front,40)
l3g = circular_cylinder(coupler_2_front,mount_2_front,40)
l4g = circular_cylinder(E_front,EF_front,40)
l5g = circular_cylinder(EF_front,F_front,70,40)

l1_front      = rigid('l1_front',l1g.mass,l1g.J,l1g.cm,l1g.C)
l2_front      = rigid('l2_front',l2g.mass,l2g.J,l2g.cm,l2g.C)
l3_front      = rigid('l3_front',l3g.mass,l3g.J,l3g.cm,l3g.C)
l4_front      = rigid('l4_front',l4g.mass,l4g.J,l4g.cm,l4g.C)
l5_front      = rigid('l5_front',l5g.mass,l5g.J,l5g.cm,l5g.C)

z=vector([0,0,1])
y=vector([0,1,0])

revA_front = revolute(mount_1_front,l1_front,chassis,z)
revD_front = revolute(mount_2_front,l3_front,chassis,z)

uniB_front = universal(coupler_1_front,l1_front,l2_front,y,-y)
uniE_front = universal(E_front,l4_front,chassis,y,-y)
uniF_front = universal(F_front,l5_front,l3_front,y,-y)

sphC_front   = spherical(coupler_2_front,l2_front,l3_front)

cylEF_front  = cylindrical(EF_front,l4_front,l5_front,y)

driver_front = absolute_locating(l5_front,'y')

##############################################
# Steering_Suspension Connection
##############################################
tie_ch_lf      = universal(tri_lf,l3_front,tie_lf,y,-y)
tie_ch_rf      = universal(tri_rf,l1_front,tie_rf,y,-y)

###############################################################################
# REAR STEARING MECHANISM
###############################################################################

mount_1_rear   = point("mount_1_rear" , [-4500  , 286  ,54+19+600])
mount_2_rear   = point("mount_2_rear" , [-4500  ,-286  ,54+19+600])
coupler_1_rear = point("C1_rear"      , [-4376  , 364  ,50+19+600])
coupler_2_rear = point("C2_rear"      , [-4376  ,-364  ,50+19+600])
E_rear         = point("E_rear"       , [-4320  , 608  ,157+19+600])
F_rear         = point("F_rear"       , [-4349  ,-285  ,85+19+600])
EF_rear        = point.mid_point(E_rear,F_rear,'EF_rear')


l1g = circular_cylinder(mount_1_rear,coupler_1_rear,40)
l2g = circular_cylinder(coupler_1_rear,coupler_2_rear,40)
l3g = circular_cylinder(coupler_2_rear,mount_2_rear,40)
l4g = circular_cylinder(E_rear,EF_rear,40)
l5g = circular_cylinder(EF_rear,F_rear,70,40)

l1_rear      = rigid('l1_rear',l1g.mass,l1g.J,l1g.cm,l1g.C)
l2_rear      = rigid('l2_rear',l2g.mass,l2g.J,l2g.cm,l2g.C)
l3_rear      = rigid('l3_rear',l3g.mass,l3g.J,l3g.cm,l3g.C)
l4_rear      = rigid('l4_rear',l4g.mass,l4g.J,l4g.cm,l4g.C)
l5_rear      = rigid('l5_rear',l5g.mass,l5g.J,l5g.cm,l5g.C)

z=vector([0,0,1])
y=vector([0,1,0])

revA_rear = revolute(mount_1_rear,l1_rear,chassis,z)
revD_rear = revolute(mount_2_rear,l3_rear,chassis,z)

uniB_rear = universal(coupler_1_rear,l1_rear,l2_rear,y,-y)
uniE_rear = universal(E_rear,l4_rear,chassis,y,-y)
uniF_rear = universal(F_rear,l5_rear,l3_rear,y,-y)

sphC_rear   = spherical(coupler_2_rear,l2_rear,l3_rear)

cylEF_rear  = cylindrical(EF_rear,l4_rear,l5_rear,y)

driver_rear = absolute_locating(l5_rear,'y')

##############################################
# Steering_Suspension Connection
##############################################
tie_ch_lr      = universal(tri_lr,l3_rear,tie_lr,y,-y)
tie_ch_rr      = universal(tri_rr,l1_rear,tie_rr,y,-y)


###############################################################################
# Defining system forces
###############################################################################
# Defining air-spring and damping properties
#############################################
deflection   = np.array([0,25,50,75,100,125,150,175,200])
spring_force = np.array([50,60,68,78,90,110,140,200,250])*1e9

velocity    = np.array([-2 ,-1.5 ,-1  ,-0.5,-0.2 ,-0.1, 0, 0.15, 0.2 ,0.3,0.5,1 ,1.5 ,2 ])*1e3
damp_force  = np.array([-25,-18  ,-15 ,-12 ,-10  ,-8  , 0, 20  , 23  ,28 ,30 ,30,48  ,60])*1e9

spring_damper_rr = air_strut('gk_w11_rr',sh_lca_rr,d1_rr,ch_sh_rr,d2_rr,[deflection,spring_force],[velocity,damp_force],80)
tf_rr            = tire_force('tvf_rr',wheel_rr,1000*1e6,1*1e6,600,vector([-3803,1100,0]))

spring_damper_lr = air_strut('gk_w11_lr',sh_lca_lr,d1_lr,ch_sh_lr,d2_lr,[deflection,spring_force],[velocity,damp_force],80)
tf_lr            = tire_force('tvf_lr',wheel_lr,1000*1e6,1*1e6,600,vector([-3803,-1100,0]))

spring_damper_rf = air_strut('gk_w11_rf',sh_lca_rf,d1_rf,ch_sh_rf,d2_rf,[deflection,spring_force],[velocity,damp_force],80)
tf_rf            = tire_force('tvf_rf',wheel_rf,1000*1e6,1*1e6,600,vector([-3803+3803,1100,0]))

spring_damper_lf = air_strut('gk_w11_lf',sh_lca_lf,d1_lf,ch_sh_lf,d2_lf,[deflection,spring_force],[velocity,damp_force],80)
tf_lf            = tire_force('tvf_lf',wheel_lf,1000*1e6,1*1e6,600,vector([-3803+3803,-1100,0]))

###############################################################################
###############################################################################
bodies_list_rr      = [uca_rr,lca_rr,upright_rr,tie_rr,d1_rr,d2_rr,wheel_rr]
bodies_list_lr      = [uca_lr,lca_lr,upright_lr,tie_lr,d1_lr,d2_lr,wheel_lr]
bodies_list_rf      = [uca_rf,lca_rf,upright_rf,tie_rf,d1_rf,d2_rf,wheel_rf]
bodies_list_lf      = [uca_lf,lca_lf,upright_lf,tie_lf,d1_lf,d2_lf,wheel_lf]
bodies_steer_rear   = [l1_rear ,l2_rear ,l3_rear ,l4_rear ,l5_rear]
bodies_steer_front  = [l1_front,l2_front,l3_front,l4_front,l5_front]
Chassis_Ground      = [chassis,ground]
bodies_list         = Chassis_Ground+bodies_list_rr+bodies_list_lr+bodies_list_rf+bodies_list_lf+bodies_steer_rear+bodies_steer_front


joints_list_rr =[uca_rev_rr,lca_rev_rr,ucao_sph_rr,lcao_sph_rr,
                tie_up_sph_rr,d2_uni_rr,d1_uni_rr,tie_ch_rr,damper_rr,wheel_hub_rr]
joints_list_lr =[uca_rev_lr,lca_rev_lr,ucao_sph_lr,lcao_sph_lr,
                tie_up_sph_lr,d2_uni_lr,d1_uni_lr,tie_ch_lr,damper_lr,wheel_hub_lr]
joints_list_rf =[uca_rev_rf,lca_rev_rf,ucao_sph_rf,lcao_sph_rf,
                tie_up_sph_rf,d2_uni_rf,d1_uni_rf,tie_ch_rf,damper_rf,wheel_hub_rf]
joints_list_lf =[uca_rev_lf,lca_rev_lf,ucao_sph_lf,lcao_sph_lf,
                tie_up_sph_lf,d2_uni_lf,d1_uni_lf,tie_ch_lf,damper_lf,wheel_hub_lf]


joints_steer_front  =[revA_front,revD_front,uniB_front,uniE_front,uniF_front,sphC_front,cylEF_front]
joints_steer_rear   =[revA_rear,revD_rear,uniB_rear,uniE_rear,uniF_rear,sphC_rear,cylEF_rear]

joints_list   = joints_list_lf+joints_list_rr+joints_list_rf+joints_list_lr+joints_steer_front+joints_steer_rear


forces    = [spring_damper_rr,spring_damper_lr,spring_damper_rf,spring_damper_lf,
               tf_rr,tf_lr,tf_rf,tf_lf]


js=pd.Series(joints_list,index=[i.name for i in joints_list])
bs=pd.Series(bodies_list,index=[i.name for i in bodies_list])
fs=pd.Series(forces     ,index=[i.name for i in forces])

wheel_drive_lf.pos=0
wheel_drive_lr.pos=0
wheel_drive_rf.pos=0
wheel_drive_rr.pos=0
driver_front.pos =l5_front.R.y
driver_rear.pos  =l5_rear.R.y
actuators = [wheel_drive_lf,wheel_drive_rf,wheel_drive_lr,wheel_drive_rr,driver_rear,driver_front]
ac=pd.Series(actuators,index=[i.name for i in actuators])


##############################################################################
# Dynamic Analysis.
##############################################################################
q0   = pd.concat([i.dic    for i in bodies_list])
qd0  = pd.concat([i.qd0()  for i in bodies_list])


topology_writer(bs,js,ac,fs,'ST100_full_dyn_datafile')

run_time=3
stepsize=0.002
arr_size= round(run_time/stepsize)


dynamic1=dds(q0,qd0,bs,js,ac,fs,'ST100_full_dyn_datafile',run_time,stepsize)
pos,vel,acc,react=dynamic1
xaxis=np.arange(0,run_time+stepsize,stepsize)

def deff(q,qdot,road):
    values=[]
    forces=[]
    for i in range(len(q)):
        forces.append(tf_lf.equation(q.loc[i],qdot.loc[i],0)[2,0])
        values.append(tf_lf.tire_deff)
    return values, forces

s=deff(pos,vel,0)

road_profile=np.zeros((arr_size+1,))

plt.figure('TDLV')
plt.subplot(211)
plt.minorticks_on()
#plt.xticks(np.linspace(0,run_time,20))
plt.plot(xaxis,np.array(s[0]),label=r'$TDLV$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.grid(which='major', linewidth='0.5', color='black')
plt.subplot(212)
#plt.xticks(np.linspace(0,run_time,20))
plt.minorticks_on()
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.grid(which='major', linewidth='0.5', color='black')
plt.show()

plt.figure('WheelCenter Position')
plt.subplot(211)
plt.plot(xaxis,pos['wheel_rr.z'],label=r'$wc RightRear_{z}$')
plt.plot(xaxis,pos['wheel_lf.z'],label=r'$wc LeftFront_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Actuator Position')
plt.subplot(211)
plt.plot(xaxis,pos['l4_rear.y'],label=r'$wc_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Half-track Change')
plt.subplot(211)
plt.plot(xaxis,pos['wheel_lf.y'],label=r'$wc_{y}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Chassis CG Vertical Position')
plt.subplot(211)
plt.plot(xaxis,pos['chassis.z'],label=r'$chassis_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('Chassis CG Lateral Position')
plt.subplot(211)
plt.plot(xaxis,pos['chassis.y'],label=r'$chassis_{y}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('WheelHub Verical Reaction Force')
plt.subplot(211)
plt.plot(xaxis,-1e-6*react['wc_lf_rev_Fz'],label=r'$wc_{Fz}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Force (N)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('UCA Mount Reaction')
plt.subplot(211)
plt.plot(xaxis,1e-6*react['ucaf_lf_rev_Fx'],label=r'$F_{x}$')
plt.plot(xaxis,1e-6*react['ucaf_lf_rev_Fy'],label=r'$F_{y}$')
plt.plot(xaxis,1e-6*react['ucaf_lf_rev_Fz'],label=r'$F_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Force (N)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

plt.figure('LCA Mount Reaction')
plt.subplot(211)
plt.plot(xaxis,1e-6*react['lcaf_lf_rev_Fx'],label=r'$F_{x}$')
plt.plot(xaxis,1e-6*react['lcaf_lf_rev_Fy'],label=r'$F_{y}$')
plt.plot(xaxis,1e-6*react['lcaf_lf_rev_Fz'],label=r'$F_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Force (N)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()


plt.figure('Shock Mount Reaction')
plt.subplot(211)
plt.plot(xaxis,1e-6*react['ch_sh_lf_uni_Fz'],label=r'$F_{z}$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Force (N)')
plt.grid()
plt.subplot(212)
plt.plot(xaxis,road_profile[0:arr_size+1],label=r'$road profile$')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()





