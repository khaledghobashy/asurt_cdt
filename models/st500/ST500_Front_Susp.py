# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:36:26 2018

@author: khaled_ghobashy
"""


from base import grf, vector, point, ep2dcm, rot2ep
from bodies_inertia import rigid, principle_inertia, thin_rod, circular_cylinder
from inertia_properties import composite_geometry, triangular_prism
from constraints import spherical, revolute, universal, \
cylindrical, rotational_drive, absolute_locating,translational
from force_elements import tsda_linear_coeff, force, tire_force
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
origin = point('origin', [0,0,0])

ch_sh  = point('ch_sh',  [-160 ,572,   471.5+546])
sh_lca = point('sh_lca', [-160 ,608,   -131.6+546])
lwr_ss = point('lwr_ss', [-160 ,604,   -61+546])

tro    = point('tro',    [279  ,763.5, 131.5+546])
tri    = point('tri',    [279  ,275,   131.5+546]) #assumed

ucaf   = point('ucaf',   [121  ,294,  140+546])
ucao   = point('ucao',   [6.5  ,777,  152+546])
ucar   = point('ucar',   [-121 ,294,  140+546])

lcaf   = point('lcaf',   [146  ,268,  -123+546])
lcao   = point('lcao',   [-4.4 ,819,  -150+546])
lcar   = point('lcar',   [-146 ,268,  -123+546])

wc     = point('wc',     [0.0  ,1032.5    , 546])
cp     = point('cp',     [0.0  ,1032.5    , 0.0])

d_m    = point.mid_point(ch_sh,sh_lca,'d_m')

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
ch_cm=vector([0,0,1300])
ch_dcm=I
ch_J=I
ch_mass  = 2015*1e3
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
cm     = vector([0,1032.5,546])
Jcm=np.array([[343952295.71, 29954.40     , -40790.37    ],
              [29954.40    , 535366217.28 , -28626.24    ],
              [-40790.37   ,-28626.24    , 343951084.62  ]])
dcm,J  = principle_inertia(Jcm)
mass   = 50*1e3  
wheel  = rigid('wheel',mass,J,cm,I)
###############################################################################

# Defining system forces
spring_damper=tsda_linear_coeff('f1',lwr_ss,d1,ch_sh,d2,k=406*1e6,lf=600,c=-40*1e6)
tf=tire_force('tvf',wheel,1000*1e6,0*1e6,546,vector([0,1032.5,0]))
#side_force=force('sf',vector([0,140*9.81*1e6,0]),upright,cp)


###############################################################################
# Defining System Joints.
###############################################################################
uca_rev     = revolute(ucaf,uca,chassis,ucaf-ucar)
lca_rev     = revolute(lcaf,lca,chassis,lcaf-lcar)
wheel_hub   = revolute(wc,wheel,upright,vector([0,-1,0]))

tie_up_sph  = spherical(tro,tie,upright)
ucao_sph    = spherical(ucao,uca,upright)
lcao_sph    = spherical(lcao,lca,upright)

damper      = cylindrical(d_m,d1,d2,lwr_ss-ch_sh)

d1_uni      = universal(sh_lca,lca, d1     ,sh_lca-lwr_ss,sh_lca-lwr_ss)
d2_uni      = universal(ch_sh ,d2 , chassis,sh_lca-lwr_ss,sh_lca-lwr_ss)
ax3         = tro-tri
tie_ch      = universal(tri,chassis,tie,vector([0,1,0]),ax3)

wheel_drive = rotational_drive(wheel_hub)

vertical_travel = absolute_locating(wheel,'z')
ch_ground       = translational(origin,ground,chassis,vector([0,0,1])) 


###############################################################################
# Collecting System Data in lists.
###############################################################################

points      =[ch_sh,ucaf,ucar,ucao,lcaf,lcar,lcao,tri,tro,cp,wc,d_m]

bodies_list =[ground,chassis,uca,lca,upright,tie,d1,d2,wheel]

joints_list =[uca_rev,lca_rev,ucao_sph,lcao_sph,
              tie_up_sph,d2_uni,d1_uni,tie_ch,damper,wheel_hub,ch_ground]

actuators = [vertical_travel,wheel_drive]
forces    = [spring_damper,tf]#,side_force]

ps=pd.Series(points     ,index=[i.name for i in points])
js=pd.Series(joints_list,index=[i.name for i in joints_list])
bs=pd.Series(bodies_list,index=[i.name for i in bodies_list])
ac=pd.Series(actuators  ,index=[i.name for i in actuators])
fs=pd.Series(forces     ,index=[i.name for i in forces])

###############################################################################
unsprung_mass = sum([i.mass for i in bodies_list[2:]])

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
actuators = [wheel_drive]
ac=pd.Series(actuators,index=[i.name for i in actuators])
    
topology_writer(bs,js,ac,fs,'ST500_dyn_datafile')

run_time=5
stepsize=0.008
arr_size= round(run_time/stepsize)

road_longitudinal = np.arange(0,1e6,20)
road_vertical     = 200*np.sin(1/5*road_longitudinal*2*np.pi*1e-3)
velocity = 20 *1e6/3600
road_profile = [max(0,ss.irregularities_height(road_longitudinal,road_vertical,velocity,i)) for i in np.arange(0,run_time+0.008,0.008) ]

#road_profile=np.concatenate([   np.zeros((round(0.5/stepsize),)),\
#                             0*np.ones ((round(1  /stepsize),)),\
#                             0*np.ones ((round(0.5  /stepsize),)),\
#                             200*np.ones ((round(1  /stepsize),)),\
#                             200*np.ones ((round(0.5  /stepsize),)),\
#                             200*np.ones ((round(1  /stepsize),)),\
#                             200*np.ones ((round(0.5  /stepsize),)),\
#                             200*np.ones ((round(1  /stepsize),)),\
#                             200*np.ones ((round(0.5  /stepsize),)),\
#                             200*np.ones ((round(1  /stepsize),)),\
#                             200*np.ones ((round(0.5  /stepsize),)),\
#                             200*np.ones ((round(1  /stepsize),)),\
#                             200*np.ones ((round(2  /stepsize),))])

#road_profile=200*np.sin(10*np.arange(0,run_time+stepsize,stepsize))

dynamic1=dds(q0,qd0,bs,js,ac,fs,'ST500_dyn_datafile',run_time,stepsize)
pos,vel,acc,react=dynamic1
xaxis=np.arange(0,run_time+stepsize,stepsize)

def deff(q,qdot,road):
    values=[]
    forces=[]
    for i in range(len(q)):
        forces.append(tf.equation(q.loc[i],qdot.loc[i],road[i])[2,0])
        values.append(tf.tire_deff)
    return values, forces

s=deff(pos,vel,road_profile)

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
plt.plot(xaxis,pos['chassis.z']-pos['chassis.z'][0]+55,label=r'$chassis_{z}$')
plt.plot(xaxis,pos['wheel.z']-pos['wheel.z'][0]+21,label=r'$wc_{z}$')
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






