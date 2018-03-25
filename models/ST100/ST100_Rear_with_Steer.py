# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:36:26 2018

@author: khaled_ghobashy
"""


from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from inertia_properties import composite_geometry, triangular_prism
from constraints import spherical, revolute, universal, \
cylindrical, rotational_drive, absolute_locating,translational,bounce_roll
from force_elements import air_strut,tsda_linear_coeff, force, tire_force
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
origin = point('origin', [-3982,0,1300])

ch_sh  = point('ch_sh',  [-3982 ,628, 1251])
sh_lca = point('sh_lca', [-3971 ,586, 508])

tro    = point('tro',    [-4217 ,788, 669])
tri    = point('tri',    [-4217 ,285, 669]) #assumed

ucaf   = point('ucaf',   [-3673 ,334, 807])
ucao   = point('ucao',   [-3803 ,812, 865])
ucar   = point('ucar',   [-3933 ,334, 807])

lcaf   = point('lcaf',   [-3463 ,269, 527])
lcao   = point('lcao',   [-3803 ,848, 453])
lcar   = point('lcar',   [-4143 ,269, 527])

wc     = point('wc',     [-3803  ,1100, 600])
cp     = point('cp',     [-3803  ,1100, 0.0])

d_m    = point.mid_point(ch_sh,sh_lca,'d_m')

###############################################################################
# Defining System Bodies and their inertia properties.
###############################################################################
I=np.eye(3)
cm=vector([-3982,0,0])
dcm=I
J=I
mass=1
ground  = rigid('ground',mass,J,cm,dcm,typ='mount')
########################################################################
I=np.eye(3)
cm=vector([-3982,0,1300])
dcm=I
J=I
mass=1
stick  = rigid('stick',mass,J,cm,dcm)
#Chassis
########
ch_cm=vector([-3803,0,1300])
ch_dcm=I
ch_J=I
ch_mass  = 2*2000*1e3
chassis  = rigid('chassis',ch_mass,ch_J,ch_cm,ch_dcm)
########################################################################
tube1    = circular_cylinder(ucaf,ucao,40,0)
tube2    = circular_cylinder(ucar,ucao,40,0)
uca_g    = composite_geometry([tube1,tube2])
uca      = rigid('uca',uca_g.mass,uca_g.J,uca_g.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf,lcao,40,0)
tube2    = circular_cylinder(lcar,lcao,40,0)
lca_g    = composite_geometry([tube1,tube2])
lca      = rigid('lca',lca_g.mass,lca_g.J,lca_g.cm,I)
########################################################################

upright_tube = circular_cylinder(lcao,ucao,60,0)
hub_cylinder = circular_cylinder(point.mid_point(lcao,ucao,"up"),wc,400,200)
upright_geo  = composite_geometry([upright_tube,hub_cylinder])

upright  = rigid('upright',upright_geo.mass,upright_geo.J,upright_geo.cm,I)
########################################################################
tie_g = circular_cylinder(tri,tro,40,0)
tie   = rigid('tie',tie_g.mass,tie_g.J,tie_g.cm,tie_g.C)
########################################################################
d1_g  = circular_cylinder(sh_lca,d_m,40)
cm    = d1_g.cm
dcm   = d1_g.C
J     = d1_g.J
mass  = d1_g.mass 
d1    = rigid('d1',mass,J,cm,dcm)
########################################################################
d2_g  = circular_cylinder(ch_sh,d_m,60,28)
cm    = d2_g.cm
dcm   = d2_g.C
J     = d2_g.J
mass  = d2_g.mass 
d2    = rigid('d2',mass,J,cm,dcm)
########################################################################
cm     = vector([-3803,1100.5,600])
Jcm=np.array([[343952295.71, 29954.40     , -40790.37    ],
              [29954.40    , 535366217.28 , -28626.24    ],
              [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 50*1e3  
wheel  = rigid('wheel',mass,J,cm,I)
###############################################################################

# Defining system forces
########################
## Defining air-spring and damping properties
#############################################
deflection   = np.array([0,25,50,75,100,125,150,175,200])
spring_force = np.array([49,50,55,60,70,80,100,125,170])*0.7*1e9

velocity    = np.array([-2 ,-1 ,-0.5,-0.1, 0, 0.1,0.15,0.18,0.2,0.5,1 ,2 , 3])*1e3
damp_force  = np.array([-55,-37,-30 ,-20 , 0, 20 ,35  ,40  ,45 ,48 ,55,60,70])*1e9

spring_damper=air_strut('gk_w11',sh_lca,d1,ch_sh,d2,[deflection,spring_force],[velocity,damp_force],80)
tf=tire_force('tvf',wheel,1000*1e6,1*1e6,600,vector([-3803,1100,0]))

###############################################################################
# Defining System Joints.
###############################################################################
uca_rev     = revolute(ucaf,uca,chassis,ucaf-ucar)
lca_rev     = revolute(lcaf,lca,chassis,lcaf-lcar)
wheel_hub   = revolute(wc,wheel,upright,vector([0,-1,0]))

tie_up_sph  = spherical(tro,tie,upright)
ucao_sph    = spherical(ucao,uca,upright)
lcao_sph    = spherical(lcao,lca,upright)

damper      = cylindrical(d_m,d1,d2,sh_lca-ch_sh)

d1_uni      = universal(sh_lca,lca, d1     ,sh_lca-d_m,sh_lca-d_m)
d2_uni      = universal(ch_sh ,d2 , chassis,sh_lca-d_m,sh_lca-d_m)
ax3         = tro-tri

wheel_drive = rotational_drive(wheel_hub)

###############################################################################

###############################################################################
# Mirror of coordinates
#######################
ch_sh_r  = point('ch_sh_r',  [-3982 ,-628, 1251])
sh_lca_r = point('sh_lca_r', [-3971 ,-586, 508])

tro_r    = point('tro_r',    [-4217 ,-788, 669])
tri_r    = point('tri_r',    [-4217 ,-285, 669]) #assumed

ucaf_r   = point('ucaf_r',   [-3673 ,-334, 807])
ucao_r   = point('ucao_r',   [-3803 ,-812, 865])
ucar_r   = point('ucar_r',   [-3933 ,-334, 807])

lcaf_r   = point('lcaf_r',   [-3463 ,-269, 527])
lcao_r   = point('lcao_r',   [-3803 ,-848, 453])
lcar_r   = point('lcar_r',   [-4143 ,-269, 527])

wc_r     = point('wc_r',     [-3803  ,-1100, 600])
cp_r     = point('cp_r',     [-3803  ,-1100, 0.0])

d_m_r    = point.mid_point(ch_sh_r,sh_lca_r,'d_m_r')

###############################################################################
# Defining System Bodies and their inertia properties.
###############################################################################
tube1    = circular_cylinder(ucaf_r,ucao_r,40,0)
tube2    = circular_cylinder(ucar_r,ucao_r,40,0)
uca_g_r    = composite_geometry([tube1,tube2])
uca_r      = rigid('uca_r',uca_g_r.mass,uca_g_r.J,uca_g_r.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf_r,lcao_r,40,0)
tube2    = circular_cylinder(lcar_r,lcao_r,40,0)
lca_g_r    = composite_geometry([tube1,tube2])
lca_r      = rigid('lca_r',lca_g_r.mass,lca_g.J,lca_g_r.cm,I)
########################################################################

upright_tube = circular_cylinder(lcao_r,ucao_r,60,0)
hub_cylinder = circular_cylinder(point.mid_point(lcao_r,ucao_r,"up_r"),wc_r,400,200)
upright_geo_r  = composite_geometry([upright_tube,hub_cylinder])

upright_r  = rigid('upright_r',upright_geo_r.mass,upright_geo_r.J,upright_geo_r.cm,I)
########################################################################
tie_g_r = circular_cylinder(tri_r,tro_r,40,0)
tie_r   = rigid('tie_r',tie_g_r.mass,tie_g_r.J,tie_g_r.cm,tie_g_r.C)
########################################################################
d1_g_r  = circular_cylinder(sh_lca_r,d_m_r,40)
cm    = d1_g_r.cm
dcm   = d1_g_r.C
J     = d1_g_r.J
mass  = d1_g_r.mass 
d1_r    = rigid('d1_r',mass,J,cm,dcm)
########################################################################
d2_g_r  = circular_cylinder(ch_sh_r,d_m_r,60,28)
cm    = d2_g_r.cm
dcm   = d2_g_r.C
J     = d2_g_r.J
mass  = d2_g_r.mass 
d2_r    = rigid('d2_r',mass,J,cm,dcm)
########################################################################
cm     = vector([-3803,-1100.5,600])
Jcm=np.array([[343952295.71, 29954.40     , -40790.37    ],
              [29954.40    , 535366217.28 , -28626.24    ],
              [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 50*1e3  
wheel_r  = rigid('wheel_r',mass,J,cm,I)
###############################################################################

# Defining system forces
###############################################################################
spring_damper_r=air_strut('gk_w11_r',sh_lca_r,d1_r,ch_sh_r,d2_r,[deflection,spring_force],[velocity,damp_force],80)
tf_r=tire_force('tvf_r',wheel_r,1000*1e6,1*1e6,600,vector([-3803,-1100,0]))

#side_right=force('sr',vector([0,0*-5*1e6*0.8*9.81*1e3,0]),upright_r,cp_r+vector([0,0,0]))
#side_left =force('sl',vector([0,0*-3*1e6*0.8*9.81*1e3,0]),upright,cp+vector([0,0,0]))

###############################################################################
# Defining System Joints.
###############################################################################
uca_rev_r     = revolute(ucaf_r,uca_r,chassis,ucaf_r-ucar_r)
lca_rev_r     = revolute(lcaf_r,lca_r,chassis,lcaf_r-lcar_r)
wheel_hub_r   = revolute(wc_r,wheel_r,upright_r,vector([0,-1,0]))

tie_up_sph_r  = spherical(tro_r,tie_r,upright_r)
ucao_sph_r    = spherical(ucao_r,uca_r,upright_r)
lcao_sph_r    = spherical(lcao_r,lca_r,upright_r)

damper_r      = cylindrical(d_m_r,d1_r,d2_r,sh_lca_r-ch_sh_r)

d1_uni_r      = universal(sh_lca_r,lca_r, d1_r     ,sh_lca_r-ch_sh_r,sh_lca_r-ch_sh_r)
d2_uni_r      = universal(ch_sh_r ,d2_r , chassis  ,sh_lca_r-ch_sh_r,sh_lca_r-ch_sh_r)
ax3_r         = tro_r-tri_r ##################################

wheel_drive_r = rotational_drive(wheel_hub_r)

vertical_travel_r = absolute_locating(wheel_r,'z')

###############################################################################
# REAR STEARING MECHANISM
###############################################################################

mount_1   = point("mount_1" , [-4500  , 286  ,54+19+600])
mount_2   = point("mount_2" , [-4500  ,-286  ,54+19+600])
coupler_1 = point("C1"      , [-4376  , 364  ,50+19+600])
coupler_2 = point("C2"      , [-4376  ,-364  ,50+19+600])
E = point("E" , [-4320  , 608  ,157+19+600])
F = point("F" , [-4349  ,-285  ,85+19+600])
EF = point.mid_point(E,F,'EF')


l1g = circular_cylinder(mount_1,coupler_1,40)
l2g = circular_cylinder(coupler_1,coupler_2,40)
l3g = circular_cylinder(coupler_2,mount_2,40)
l4g = circular_cylinder(E,EF,40)
l5g = circular_cylinder(EF,F,70,40)

l1      = rigid('l1',l1g.mass,l1g.J,l1g.cm,l1g.C)
l2      = rigid('l2',l2g.mass,l2g.J,l2g.cm,l2g.C)
l3      = rigid('l3',l3g.mass,l3g.J,l3g.cm,l3g.C)
l4      = rigid('l4',l4g.mass,l4g.J,l4g.cm,l4g.C)
l5      = rigid('l5',l5g.mass,l5g.J,l5g.cm,l5g.C)

z=vector([0,0,1])
y=vector([0,1,0])

revA = revolute(mount_1,l1,chassis,z)
revD = revolute(mount_2,l3,chassis,z)

uniB = universal(coupler_1,l1,l2,y,-y)
uniE = universal(E,l4,chassis,y,-y)
uniF = universal(F,l5,l3,y,-y)

sphC = spherical(coupler_2,l2,l3)

cylEF = cylindrical(EF,l4,l5,y)

driver= absolute_locating(l5,'y')

##############################################
# Steering_Suspension Connection
##############################################
tie_ch_r      = universal(tri_r,l3,tie_r,y,-y)
tie_ch        = universal(tri,l1,tie,y,-y)

##############################################
# chassis ground connection
##############################################
ch_stick = revolute(origin,chassis,stick,vector([1,0,0]))
gr_stick = translational(origin,ground,stick,vector([0,0,1]))

grounded = bounce_roll(origin,ground,chassis,vector([0,0,1]),vector([1,0,0]))

###############################################################################
# Collecting System Data in lists.
###############################################################################

points      =[ch_sh,ucaf,ucar,ucao,lcaf,lcar,lcao,tri,tro,cp,wc,d_m]

bodies_list_l =[ground,chassis,uca,lca,upright,tie,d1,d2,wheel]
bodies_list_r =[uca_r,lca_r,upright_r,tie_r,d1_r,d2_r,wheel_r]
bodies_steer  =[l1,l2,l3,l4,l5]
bodies_list   = bodies_list_l+bodies_list_r+bodies_steer


joints_list_l =[uca_rev,lca_rev,ucao_sph,lcao_sph,
              tie_up_sph,d2_uni,d1_uni,tie_ch,damper,wheel_hub]
joints_list_r =[uca_rev_r,lca_rev_r,ucao_sph_r,lcao_sph_r,
              tie_up_sph_r,d2_uni_r,d1_uni_r,tie_ch_r,damper_r,wheel_hub_r]
joints_steer  =[revA,revD,uniB,uniE,uniF,sphC,cylEF]
joints_list   = joints_list_l+joints_list_r+joints_steer+[grounded]

actuators_l = [vertical_travel,wheel_drive]
actuators_r = [vertical_travel_r,wheel_drive_r]
actuators_s = [driver]
actuators   = actuators_l+actuators_r+actuators_s

forces_l    = [spring_damper,tf]
forces_r    = [spring_damper_r,tf_r]#,side_right,side_left]
forces      = forces_l+forces_r


ps=pd.Series(points     ,index=[i.name for i in points])
js=pd.Series(joints_list,index=[i.name for i in joints_list])
bs=pd.Series(bodies_list,index=[i.name for i in bodies_list])
ac=pd.Series(actuators  ,index=[i.name for i in actuators])
fs=pd.Series(forces     ,index=[i.name for i in forces])


##############################################################################
# Kinematically driven analysis.
##############################################################################
#topology_writer(bs,js,ac,fs,'ST500_datafile')
#q0   = pd.concat([i.dic    for i in bodies_list])
#time=np.linspace(0,2*np.pi,200)
#wheel_drive.pos_array=np.zeros((len(time),))
#vertical_travel.pos_array=546+180*np.sin(time)
#
#kds_run=kds(bs,js,ac,'ST500_datafile',time)
#kds_reactions=reactions(kds_run[0],kds_run[1],kds_run[2],bs,js,ac,fs,'asurt18_kds_datafile')
#
#plt.figure('Half-track Change')
#plt.plot(kds_run[0]['wheel.z'],kds_run[0]['wheel.y'],label=r'$wc_{y}$')
#plt.legend()
#plt.xlabel('Vertical Travel (mm)')
#plt.ylabel('Displacement (mm)')
#plt.grid()
#plt.show()
#
#plt.figure('Shock Mount Reaction')
#plt.plot(kds_run[0]['wheel.z'],1e-6*kds_reactions[5]['ch_sh_uni_Fz'],label=r'$F_{z}$')
#plt.legend()
#plt.xlabel('Vertical Travel (mm)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()
#
#plt.figure('WheelHub Verical Reaction Force')
#plt.plot(kds_run[0]['wheel.z'],-1e-6*kds_reactions[5]['wc_rev_Fz'],label=r'$wc_{Fz}$')
#plt.legend()
#plt.xlabel('Vertical Travel (mm)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()


##############################################################################
# Dynamic Analysis.
##############################################################################
q0   = pd.concat([i.dic    for i in bodies_list])
qd0  = pd.concat([i.qd0()  for i in bodies_list])

wheel_drive.pos=0
wheel_drive_r.pos=0
driver.pos=l5.R.y
actuators = [wheel_drive,wheel_drive_r,driver]
ac=pd.Series(actuators,index=[i.name for i in actuators])
    
topology_writer(bs,js,ac,fs,'ST100_dyn_datafile')

run_time=5
stepsize=0.002
arr_size= round(run_time/stepsize)


dynamic1=dds(q0,qd0,bs,js,ac,fs,'ST100_dyn_datafile',run_time,stepsize)
pos,vel,acc,react=dynamic1
xaxis=np.arange(0,run_time+stepsize,stepsize)

def deff(q,qdot,road):
    values=[]
    forces=[]
    for i in range(len(q)):
        forces.append(tf.equation(q.loc[i],qdot.loc[i],0)[2,0])
        values.append(tf.tire_deff)
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
plt.plot(xaxis,pos['wheel.z'],label=r'$wc_{z}$')
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
plt.plot(xaxis,pos['l4.y'],label=r'$wc_{z}$')
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
plt.plot(xaxis,pos['wheel.y'],label=r'$wc_{y}$')
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

plt.figure('WheelHub Verical Reaction Force')
plt.subplot(211)
plt.plot(xaxis,-1e-6*react['wc_rev_Fz'],label=r'$wc_{Fz}$')
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
plt.plot(xaxis,1e-6*react['ucaf_rev_Fx'],label=r'$F_{x}$')
plt.plot(xaxis,1e-6*react['ucaf_rev_Fy'],label=r'$F_{y}$')
plt.plot(xaxis,1e-6*react['ucaf_rev_Fz'],label=r'$F_{z}$')
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
plt.plot(xaxis,1e-6*react['lcaf_rev_Fx'],label=r'$F_{x}$')
plt.plot(xaxis,1e-6*react['lcaf_rev_Fy'],label=r'$F_{y}$')
plt.plot(xaxis,1e-6*react['lcaf_rev_Fz'],label=r'$F_{z}$')
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
plt.plot(xaxis,1e-6*react['ch_sh_uni_Fz'],label=r'$F_{z}$')
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


plt.figure('Steering Actuator Fixation')
plt.subplot(211)
plt.plot(xaxis,1e-6*react['E_uni_Fy'],label=r'$F_{y}$')
plt.plot(xaxis,1e-6*react['E_uni_Fx'],label=r'$F_{x}$')
plt.plot(xaxis,1e-6*react['E_uni_Fz'],label=r'$F_{z}$')
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


plt.figure('Roll Joint')
plt.subplot(211)
plt.plot(xaxis,1e-6*react['origin_br_Fy'],label=r'$F_{y}$')
plt.plot(xaxis,1e-6*react['E_uni_Fx'],label=r'$F_{x}$')
plt.plot(xaxis,1e-6*react['E_uni_Fz'],label=r'$F_{z}$')
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

