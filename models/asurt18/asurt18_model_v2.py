# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:32:37 2017

@author: khale
"""

from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from inertia_properties import composite_geometry, triangular_prism
from constraints import spherical, revolute, universal, \
cylindrical, rotational_drive, absolute_locating,translational
from force_elements import tsda, force, tire_force
from pre_processor import topology_writer
import pandas as pd
import numpy as np
from solvers import kds, check_jacobian_dense, reactions, dds, state_space_creator
from newton_raphson import nr_kds
import matplotlib.pyplot as plt

###############################################################################
# Defining System HardPoints.
###############################################################################
origin = point('origin', [0,0,0])
bcp    = point('bcp',    [-33.403, 289.196, 175])
bc_pr  = point('bc_pr',  [-32.319, 349.182, 175.695])
bc_sh  = point('bc_sh',  [-26.425, 322.818, 215.371])
cp     = point('cp',     [0.0    , 585    , 0.0])
lcaf   = point('lcaf',   [142.669, 262.329, 127.709])
lcao   = point('lcao',   [9.741  , 494.954, 144.5])
lcar   = point('lcar',   [-149   , 285.237, 173.602])
uca_pr = point('uca_pr', [-12.927, 451.551, 286.974])
ch_sh  = point('ch_sh',  [10     , 263.651, 450])
tro    = point('tro',    [75     , 487.598, 181.183])
tri    = point('tri',    [75     , 267.214, 180])
ucaf   = point('ucaf',   [140.893, 260.927, 277.539])
ucao   = point('ucao',   [-9.741 , 488.982, 315.5])
ucar   = point('ucar',   [-150.08, 286.272, 310.841])
wc     = point('wc',     [0.0    , 585    , 230])
d_m    = point.mid_point(ch_sh,bc_sh,'d_m')

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
ch_cm=vector([0,0,0])
ch_dcm=I
ch_J=I
ch_mass  = 70*1e3
chassis  = rigid('chassis',ch_mass,ch_J,ch_cm,ch_dcm,typ='mount')
########################################################################
tube1    = circular_cylinder(ucaf,ucao,10,8)
tube2    = circular_cylinder(ucar,ucao,10,8)
uca_g    = composite_geometry([tube1,tube2])
uca      = rigid('uca',uca_g.mass,uca_g.J,uca_g.cm,I)
########################################################################
tube1    = circular_cylinder(lcaf,lcao,10,8)
tube2    = circular_cylinder(lcar,lcao,10,8)
lca_g    = composite_geometry([tube1,tube2])
lca      = rigid('lca',lca_g.mass,lca_g.J,lca_g.cm,I)
########################################################################
cm=vector([-5.21,530.71,239.46])
Jcm=np.array([[6809559.26,-70112.53, 723753.00],
              [-70112.53, 6663047.19,-111547.75],
              [723753.00,-111547.75,1658347.31]])
dcm,J=principle_inertia(Jcm)
mass = 1329.83 
upright  = rigid('upright',mass,J,cm,dcm)
########################################################################
rocker_g = triangular_prism(bcp,bc_sh,bc_pr,34,2.7)
rocker   = rigid('rocker',rocker_g.mass,rocker_g.J,rocker_g.cm,rocker_g.C)
########################################################################
push_g = circular_cylinder(bc_pr,uca_pr,12,8)  
push   = rigid('push',push_g.mass,push_g.J,push_g.cm,push_g.C)
########################################################################
tie_g = circular_cylinder(tri,tro,12,8)
tie   = rigid('tie',tie_g.mass,tie_g.J,tie_g.cm,tie_g.C)
########################################################################
d1_g  = circular_cylinder(bc_sh,d_m,15)
cm    = d1_g.cm
dcm   = d1_g.C
J     = d1_g.J
mass  = d1_g.mass 
d1    = rigid('d1',mass,J,cm,dcm)
########################################################################
d2_g  = circular_cylinder(ch_sh,d_m,35,28)
cm    = d2_g.cm
dcm   = d2_g.C
J     = d2_g.J
mass  = d2_g.mass 
d2    = rigid('d2',mass,J,cm,dcm)
########################################################################
cm     = vector([0,613.93,230])
Jcm=np.array([[343952295.71, 29954.40     , -40790.37    ],
              [29954.40    , 535366217.28 , -28626.24    ],
              [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 4*1e3  
wheel  = rigid('wheel',mass,J,cm,I)
###############################################################################

# Defining system forces
seat1=bc_sh+(50*(ch_sh-bc_sh).unit)
seat2=bc_sh+(170*(ch_sh-bc_sh).unit)
spring_damper=tsda('f1',seat1,d1,seat2,d2,k=90*1e6,lf=135,c=-8*1e6)
#nl=(160)*9.81*1e6
#force_vector=np.array([[nl*1],[nl*1],[0]])
#vf=force('vertical_force',force_vector,wheel,vector([0,-600,0]))
tf=tire_force('tvf',wheel,300*1e6,-1*1e6,230,vector([0,585,0]))
side_force=force('sf',vector([0,140*9.81*1e6,0]),upright,cp)


###############################################################################
# Defining System Joints.
###############################################################################
uca_rev     = revolute(ucaf,uca,chassis,ucaf-ucar)
lca_rev     = revolute(lcaf,lca,chassis,lcaf-lcar)
bcp_rev     = revolute(bcp,rocker,chassis,vector.normal(bc_pr,bc_sh,bcp,grf))
wheel_hub   = revolute(wc,wheel,upright,vector([0,-1,0]))

pr_uca_sph  = spherical(uca_pr,uca,push)
tie_up_sph  = spherical(tro,tie,upright)
ucao_sph    = spherical(ucao,uca,upright)
lcao_sph    = spherical(lcao,lca,upright)

damper      = cylindrical(d_m,d1,d2,bc_sh-ch_sh)

ax1         = uca_pr-bc_pr
pr_bc       = universal(bc_pr,rocker,push,ax1,bc_pr-bcp)
ax2         = bc_sh-ch_sh
sh_bc       = universal(bc_sh,rocker,d1,ax2,bc_sh-bcp)
d2_ch_uni   = universal(ch_sh,d2,chassis,ax2,-ax2)
ax3         = tro-tri
tie_ch      = universal(tri,chassis,tie,vector([0,1,0]),ax3)

wheel_drive = rotational_drive(wheel_hub)

vertical_travel = absolute_locating(wheel,'z')
ch_ground       = translational(origin,ground,chassis,vector([0,0,1])) 


###############################################################################
# Collecting System Data in lists.
###############################################################################

points      =[bcp,bc_sh,bc_pr,ch_sh,ucaf,ucar,ucao,lcaf,lcar,lcao,tri,tro,uca_pr,cp,wc,d_m]

bodies_list =[chassis,uca,lca,upright,push,tie,d1,d2,rocker,wheel]

joints_list =[uca_rev,lca_rev,bcp_rev,ucao_sph,lcao_sph,pr_uca_sph,
              tie_up_sph,d2_ch_uni,sh_bc,tie_ch,pr_bc,damper,wheel_hub]

actuators = [vertical_travel,wheel_drive]
forces    = [spring_damper,tf]#,side_force]

ps=pd.Series(points     ,index=[i.name for i in points])
js=pd.Series(joints_list,index=[i.name for i in joints_list])
bs=pd.Series(bodies_list,index=[i.name for i in bodies_list])
ac=pd.Series(actuators  ,index=[i.name for i in actuators])
fs=pd.Series(forces     ,index=[i.name for i in forces])

##############################################################################
# Kinematically driven analysis.
##############################################################################
topology_writer(bs,js,ac,fs,'asurt18_kds_datafile')
q0   = pd.concat([i.dic    for i in bodies_list])
time=np.linspace(0,np.pi,100)
wheel_drive.pos_array=np.zeros((len(time),))
vertical_travel.pos_array=230+30*np.sin(2*time)

kds_run=kds(bs,js,ac,'asurt18_kds_datafile',time)
kds_reactions=reactions(kds_run[0],kds_run[1],kds_run[2],bs,js,ac,fs,'asurt18_kds_datafile')

plt.figure('WheelHub Verical Reaction Force')
plt.plot(kds_run[0]['wheel.z'],-1e-6*kds_reactions[5]['wc_rev_Fz'],label=r'$wc_{Fz}$')
plt.legend()
plt.xlabel('Vertical Travel (mm)')
plt.ylabel('Force (N)')
plt.grid()
plt.show()


##############################################################################
# Dynamic Analysis.
##############################################################################
#q0   = pd.concat([i.dic    for i in bodies_list])
#qd0  = pd.concat([i.qd0()  for i in bodies_list])
#qdd0 = pd.concat([i.qdd0() for i in bodies_list])
#
#vertical_travel=absolute_locating(wheel,'z')
#wheel_drive.pos=0
#vertical_travel.pos=230
#actuators = [wheel_drive]
#ac=pd.Series(actuators,index=[i.name for i in actuators])
#    
#topology_writer(bs,js,ac,fs,'asurt18_datafile')
#
#run_time=0.5
#stepsize=0.0025
#
#dynamic1=dds(q0,qd0,bs,js,ac,fs,'asurt18_datafile',run_time,stepsize)
#pos,vel,acc,react=dynamic1
#xaxis=np.arange(0,run_time+stepsize,stepsize)
#
#plt.figure('WheelCenter Position')
#plt.plot(xaxis,pos['wheel.z'],label=r'$wc_{z}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Displacement (mm)')
#plt.grid()
#plt.show()
#
#plt.figure('Half-track Change')
#plt.plot(xaxis,pos['wheel.y'],label=r'$wc_{y}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Displacement (mm)')
#plt.grid()
#plt.show()
#
#plt.figure('Chassis CG Vertical Position')
#plt.plot(xaxis,pos['chassis.z'],label=r'$chassis_{z}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Displacement (mm)')
#plt.grid()
#plt.show()
#
#plt.figure('WheelHub Verical Reaction Force')
#plt.plot(xaxis,-1e-6*react['wc_rev_Fz'],label=r'$wc_{Fz}$')
#plt.plot(xaxis,-1e-6*react['wc_rev_Fx'],label=r'$wc_{Fx}$')
#plt.plot(xaxis,-1e-6*react['wc_rev_Fy'],label=r'$wc_{Fy}$')
#plt.plot(xaxis,-1e-9*react['wc_rev_Mx'],label=r'$M_{x}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()
#
#plt.figure('UCA Mount Reaction')
#plt.plot(xaxis,1e-6*react['ucaf_rev_Fx'],label=r'$F_{x}$')
#plt.plot(xaxis,1e-6*react['ucaf_rev_Fy'],label=r'$F_{y}$')
#plt.plot(xaxis,1e-6*react['ucaf_rev_Fz'],label=r'$F_{z}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()
#
#plt.figure('LCA Mount Reaction')
#plt.plot(xaxis,1e-6*react['lcaf_rev_Fx'],label=r'$F_{x}$')
#plt.plot(xaxis,1e-6*react['lcaf_rev_Fy'],label=r'$F_{y}$')
#plt.plot(xaxis,1e-6*react['lcaf_rev_Fz'],label=r'$F_{z}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()
#
#plt.figure('Tie_Chassis Mount Reaction')
#plt.plot(xaxis,1e-6*react['tri_uni_Fx'],label=r'$F_{x}$')
#plt.plot(xaxis,1e-6*react['tri_uni_Fy'],label=r'$F_{y}$')
#plt.plot(xaxis,1e-6*react['tri_uni_Fz'],label=r'$F_{z}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()
#
#plt.figure('Shock Mount Reaction')
#plt.plot(xaxis,1e-6*react['ch_sh_uni_Fx'],label=r'$F_{x}$')
#plt.plot(xaxis,-1e-6*react['ch_sh_uni_Fy'],label=r'$F_{y}$')
#plt.plot(xaxis,1e-6*react['ch_sh_uni_Fz'],label=r'$F_{z}$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('Force (N)')
#plt.grid()
#plt.show()






