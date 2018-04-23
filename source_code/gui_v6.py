# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:15:03 2018

@author: khaled.ghobashy
"""


import ipywidgets as widgets
import networkx as nx
import matplotlib.pyplot as plt
import IPython as ipy
import qgrid
import time
import pandas as pd
from base import point, vector
from bodies_inertia import rigid
from inertia_properties import circular_cylinder
from force_elements import air_strut
from solvers import kds, reactions
from pre_processor import topology_writer
import numpy as np
from constraints import cylindrical, spherical, revolute, translational,\
                        universal, rotational_drive, absolute_locating

layout80px  = widgets.Layout(width='80px')
layout100px = widgets.Layout(width='100px')
layout120px = widgets.Layout(width='120px')
layout200px = widgets.Layout(width='200px')

separator100      = widgets.Label(value=''.join(75*['_']))
separator50       = widgets.Label(value=''.join(50*['_']))
vertical_space = widgets.Label(value='\n')

def openfile_dialog():
    from PyQt5 import QtGui
    from PyQt5 import QtGui, QtWidgets
    app = QtWidgets.QApplication([dir])
    fname = QtWidgets.QFileDialog.getOpenFileName(None, "Select a file...", '.', filter="All files (*)")
    return fname[0]

def savefile_dialog():
    from PyQt5 import QtGui
    from PyQt5 import QtGui, QtWidgets
    app = QtWidgets.QApplication([dir])
    fname = QtWidgets.QFileDialog.getSaveFileName(None, "Save File", '.', filter="All files (*)")
    return fname[0]







class model(object):
    
    def __init__(self):
        self.tab       = widgets.Tab()
        
        self.points     = pd.Series()
        self.bodies     = pd.Series()
        self.joints     = pd.Series()
        self.geometries = pd.Series()
        self.vectors    = pd.Series()
        self.forces     = pd.Series()
        self.model      = pd.Series()
        
        self.name = ''
        
        self.out = widgets.Output()
        
        self.points_dataframe = pd.DataFrame(columns=['x','y','z','Alignment','Notes'])
        self.bodies_dataframe = pd.DataFrame(columns=['mass','Rx','Ry','Rz',
                                                      'Ixx','Iyx','Izx',
                                                      'Ixy','Iyy','Izy',
                                                      'Ixz','Iyz','Izz',
                                                      'xx','yx','zx',
                                                      'xy','yy','zy',
                                                      'xz','yz','zz'])
    
        self.topology  = nx.Graph()
        self.data_flow = nx.DiGraph()

    
    
    def _filter_points(self):
        return dict(pd.concat([self.points.filter(like='hpr_'),self.points.filter(like='hps_')]))
    def _filter_bodies(self):
        return dict(pd.concat([self.bodies.filter(like='rbr_'),self.bodies.filter(like='rbs_')]))
    
    def _sort(self):
        self.points=self.points.sort_index()
        self.bodies=self.bodies.sort_index()
        self.joints=self.joints.sort_index()
        self.vectors=self.vectors.sort_index()
        self.geometries=self.geometries.sort_index()
        self.forces=self.forces.sort_index()
        
                        
    def save_model_copy(self):
        
        save_button = widgets.Button(description=' Save as',tooltip='Save Copy of the Model and open')
        save_button.icon='copy'
        save_button.layout=layout100px
        def save_click(dummy):
            with self.out:
                f=savefile_dialog()
                if f=='':
                    return
                self._sort()
                self.model['points']=self.points
                self.model['bodies']=self.bodies
                self.model['joints']=self.joints
                self.model['geometries']=self.geometries
                self.model['vectors']=self.vectors
                self.model['forces']=self.forces
                self.model['data_flow']=self.data_flow
                
                self.model.to_pickle(f)
                print('Model Saved as "%s" at %s'%(f.split("/")[-1],time.strftime('%I:%M:%S %p')))
        save_button.on_click(save_click)
        
        return widgets.VBox([save_button])
    
    def save_model(self):
        
        save_button = widgets.Button(description=' Save',tooltip='Save Model')
        save_button.icon='save'
        save_button.layout=layout100px
        def save_click(dummy):
            with self.out:
                self._sort()
                self.model['points']=self.points
                self.model['bodies']=self.bodies
                self.model['joints']=self.joints
                self.model['geometries']=self.geometries
                self.model['vectors']=self.vectors
                self.model['forces']=self.forces
                self.model['data_flow']=self.data_flow

                
                self.model.to_pickle(self.name)
                print('Model Saved as "%s" at %s'%(self.name.split("/")[-1],time.strftime('%I:%M:%S %p')))
        save_button.on_click(save_click)
        
        return widgets.VBox([save_button])
                
    def open_model(self):
        
        open_button = widgets.Button(description=' Open',tooltip='Open Model Binary files')
        open_button.icon='folder-open'
        open_button.layout=layout100px
        def open_click(dummy):
            self.out.clear_output()
            with self.out:
                f=openfile_dialog()
                if f=='':
                    return
                self.name=f
                name_l = widgets.HTML('<b>'+self.name.split('/')[-1])

                self.model=pd.read_pickle(f)
                self.points,self.bodies,self.joints,self.geometries,self.vectors,self.forces,self.data_flow=self.model
                self._sort()
                for i in self.points:
                    self.points_dataframe.loc[i.name]=[i.x,i.y,i.z,i.alignment,i.notes]
                
                
                modeling = widgets.Accordion()
                modeling.children=[self.add_point(),
                                 self.add_vectors(),
                                 self.add_bodies(),
                                 self.add_joints(),
                                 self.add_actuators(),
                                 self.add_forces()]
                
                modeling.set_title(0,'SYSTEM POINTS')
                modeling.set_title(1,'SYSTEM MARKERS')
                modeling.set_title(2,'SYSTEM BODIES')
                modeling.set_title(3,'SYSTEM JOINTS')
                modeling.set_title(4,'SYSTEM ACTUATORS')
                modeling.set_title(5,'SYSTEM FORCES')
                
#                simulation = widgets.Accordion(children=[self.parallel_travel()])
#                simulation.set_title(0,'PARALLEL WHEEL TRAVEL')
                
                
                post_processing = self.data_processing()
                
                major_fields = widgets.Accordion(children=[modeling,post_processing])
                major_fields.set_title(0,'MODELING')
                major_fields.set_title(1,'SIMULATION')
                major_fields.set_title(2,'POST PROCESSING')
                
                
                print('Model "%s" Loaded at %s'%(f.split("/")[-1],time.strftime('%I:%M:%S %p')))
                return ipy.display.display(widgets.VBox([name_l,major_fields]))
                
        open_button.on_click(open_click)
        
        
        return widgets.VBox([open_button])
    
    def new_model(self):
        
        new_button = widgets.Button(description=' New',tooltip='Creat a New Model')
        new_button.icon='file'
        new_button.layout=layout100px
        def new_click(dummy):
            self.out.clear_output()
            with self.out:
                f=savefile_dialog()
                if f=='':
                    return
                self.name=f.split("/")[-1]
                self._sort()
                
                self.model['points']=self.points
                self.model['bodies']=self.bodies
                self.model['joints']=self.joints
                self.model['geometries']=self.geometries
                self.model['vectors']=self.vectors
                self.model['forces']=self.forces
                self.model['data_flow']=self.data_flow
                
                self.model.to_pickle(f)
                
                fields = widgets.Accordion()
                fields.children=[self.add_point(),self.add_vectors(),self.add_bodies(),self.add_joints(),
                                 self.add_forces()]
                fields.set_title(0,'SYSTEM POINTS')
                fields.set_title(1,'SYSTEM MARKERS')
                fields.set_title(2,'SYSTEM BODIES')
                fields.set_title(3,'SYSTEM JOINTS')
                fields.set_title(4,'SYSTEM FORCES')
                print('New Model "%s" Created at %s'%(self.name,time.strftime('%I:%M:%S %p')))
                return ipy.display.display(fields)
                
        new_button.on_click(new_click)
        
        return widgets.VBox([new_button])
        
    def add_point(self):
        
        tabs = widgets.Tab()
        tabs.set_title(0,'ADD NEW POINT')
        tabs.set_title(1,'POINTS TABLE')
        tabs.set_title(2,'IMPORT / EXPORT')
        
        tab1_out = widgets.Output()
        tab2_out = widgets.Output()
        
        tabel=qgrid.QgridWidget(df=self.points_dataframe)
        with tab2_out:
            ipy.display.display(tabel)
        
        name_l = widgets.HTML('<b>Name')
        name_v = widgets.Text(placeholder='Point Name')
        name_b = widgets.VBox([name_l,name_v])
        
        x_l = widgets.Label(value='$x$')
        x_v = widgets.FloatText()
        x_b = widgets.VBox([x_l,x_v])
        
        y_l = widgets.Label(value='$y$')
        y_v = widgets.FloatText()
        y_b = widgets.VBox([y_l,y_v])
        
        z_l = widgets.Label(value='$z$')
        z_v = widgets.FloatText()
        z_b = widgets.VBox([z_l,z_v])
    
        name_l.layout=layout100px
        name_v.layout=layout120px
        x_l.layout=y_l.layout=z_l.layout=layout100px
        x_v.layout=y_v.layout=z_v.layout=layout100px
        
        notes_l = widgets.HTML('<b>Notes')        
        notes_v = widgets.Textarea(placeholder='Optional brief note/description.')
        notes_v.layout=widgets.Layout(width='350px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'hpr_','L':'hpl_','S':'hps_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])        
        edit_l = widgets.HTML('<b>Edit Point')
        
        points_dropdown = widgets.Dropdown()
        points_dropdown.options = dict(self.points)
        points_dropdown.layout=layout120px
        
        
        
        add_button = widgets.Button(description='Apply',icon='check')
        add_button.layout=layout100px
        def add_click(dummy):
            tab1_out.clear_output()
            tab2_out.clear_output()
            with tab1_out:
                name,x,y,z = [i.value for i in [name_v,x_v,y_v,z_v]]
                
                if name=='':
                    print('Please enter a valid name in the Point Name field')
                    return 
                
                if (alignment_v.label=='R' and y<0) or (alignment_v.label=='L' and y>0):
                    print('Inconsistent Selction of y value and symmetry!!')
                    return

                name=alignment_v.value+name
                
                if alignment_v.label!='S':
                    p1=point(name,[x,y,z])
                    p1.alignment=alignment_v.label
                    p2 = p1.m_object
                    p1.notes=p2.notes=notes_v.value
                    
                    self.points[p1.name]=p1
                    self.points[p2.name]=p2
                    
                    self.points_dataframe.loc[p1.name]=[x,p1.y,z,p1.alignment,notes_v.value]
                    self.points_dataframe.loc[p2.name]=[x,p2.y,z,p2.alignment,notes_v.value]
                    
                    self.data_flow.add_node(p1.name,obj=p1)
                    self.data_flow.add_node(p2.name,obj=p2)
    
                else:
                    p=point(name,[x,y,z])
                    p.alignment=alignment_v.label
                    p.notes=notes_v.value
                    self.points[p.name]=p
                    self.points_dataframe.loc[p.name]=[x,y,z,p.alignment,notes_v.value]
                    self.data_flow.add_node(p.name,obj=p)
                    
                
                name_v.value=''
                notes_v.value=''
                self._sort()
                points_dropdown.options=dict(self.points)
                
            with tab2_out:
                tabel.df=self.points_dataframe
                ipy.display.display(tabel)
        add_button.on_click(add_click)
        
        
        
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                with tab1_out:
                    if points_dropdown.label==None:
                        return
                    
                    name_v.value=points_dropdown.label[4:]
                    x_v.value=points_dropdown.value.x
                    y_v.value=points_dropdown.value.y
                    z_v.value=points_dropdown.value.z
                    alignment_v.label=points_dropdown.value.alignment
                    notes_v.value=points_dropdown.value.notes
        points_dropdown.observe(on_change)
        
        
        edit_button = widgets.Button(description='Edit',icon='edit',tooltip='apply edits to selected point')
        edit_button.layout=layout100px
        def edit_click(dummy):
            tab1_out.clear_output()
            
            with tab1_out:
                dependencies = nx.DiGraph(nx.edge_dfs(self.data_flow,points_dropdown.label))
                nx.draw_circular(nx.DiGraph(dependencies),with_labels=True)
                plt.show()
            
            with tab1_out:
                name,x,y,z = [i.value for i in [name_v,x_v,y_v,z_v]]
                
                if (alignment_v.label=='R' and y<0) or (alignment_v.label=='L' and y>0):
                    print('Inconsistent Selction of y value and symmetry!!')
                    return
                
                if alignment_v.label in 'RL':
                    self.points[points_dropdown.label].x=x
                    self.points[points_dropdown.label].y=y
                    self.points[points_dropdown.label].z=z
                    
                    self.points[points_dropdown.value.m_name].x=x
                    self.points[points_dropdown.value.m_name].y=-y
                    self.points[points_dropdown.value.m_name].z=z
                    
                    for e in nx.edge_dfs(self.data_flow,points_dropdown.label):
                       
                        try:
                            self.data_flow.node[e[1]]['obj'].__setattr__(self.data_flow.edges[e]['attr'],self.data_flow.node[e[0]]['obj'])
                            
                        except KeyError:
                            print('Not Found \n')
                            pass
                    
                else:
                    self.points[points_dropdown.label].x=x
                    self.points[points_dropdown.label].y=y
                    self.points[points_dropdown.label].z=z
        edit_button.on_click(edit_click)
        
        
        field1 = widgets.HBox([name_b,x_b,y_b,z_b])
        field2 = widgets.VBox([alignment_b,notes_b])
        field3 = widgets.VBox([edit_l,points_dropdown,edit_button])

        
        tab1_content = widgets.VBox([field1,field2,add_button,separator100,field3,tab1_out])
        tab2_content = widgets.VBox([tab2_out],layout=widgets.Layout(width='550px'))
        
        tabs.children=[tab1_content,tab2_content]
        
        return tabs
    
    
    def add_vectors(self):

        vectors_out = widgets.Output()
        vectors_sub = widgets.Output()
                
        name_l = widgets.HTML('<b>Vector Name')
        name_v = widgets.Text(placeholder='vector name here')
        name_b = widgets.VBox([name_l,name_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'ovr_','L':'ovl_','S':'ovs_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])
        
        methods  = ('User Entered Value','Relative Position','Normal to Plane')
        method_l = widgets.HTML('<b> Creation Method')
        method_v = widgets.Dropdown(options=methods)
        method_b = widgets.VBox([method_l,method_v])
        
        notes_l = widgets.HTML('<b>Notes')        
        notes_v = widgets.Textarea(placeholder='Optional brief note/description.')
        notes_v.layout=widgets.Layout(width='350px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        main_data_block = widgets.VBox([name_b,alignment_b,notes_b,separator50,method_b])
        
#        vectors_dropdown_l = widgets.HTML('<b> Select Vector')
#        vectors_dropdown_v = widgets.Dropdown(options=self._filter)
        #######################################################################
        # Creating Data for first method.
        #######################################################################
        
        x_l = widgets.Label(value='$x$')
        x_v = widgets.FloatText()
        x_b = widgets.VBox([x_l,x_v])
        
        y_l = widgets.Label(value='$y$')
        y_v = widgets.FloatText()
        y_b = widgets.VBox([y_l,y_v])
        
        z_l = widgets.Label(value='$z$')
        z_v = widgets.FloatText()
        z_b = widgets.VBox([z_l,z_v])
    
        x_l.layout=y_l.layout=z_l.layout=layout100px
        x_v.layout=y_v.layout=z_v.layout=layout100px
                
        m1_block = widgets.VBox([widgets.HBox([x_b,y_b,z_b]),vectors_sub])
        
        m1_button = widgets.Button(description='Apply',icon='check')
        m1_button.layout=layout100px
        def m1_click(dummy):
            with vectors_sub:
                name,x,y,z = [i.value for i in [name_v,x_v,y_v,z_v]]
                
                if name=='':
                    print('Please enter a valid name in the Vector Name field')
                    return 
                
                if (alignment_v.label=='R' and y<0) or (alignment_v.label=='L' and y>0):
                    print('Inconsistent Selction of y value and symmetry!!')
                    return

                name=alignment_v.value+name
                
                if alignment_v.label!='S':
                    v1=point(name,[x,y,z])
                    v1.alignment=alignment_v.label
                    v2 = v1.m_object
                    v2.name=v2.name.replace('hp','ov')
                    self.vectors[v1.name]=v1
                    self.vectors[v2.name]=v2
                    
                    self.data_flow.add_node(v1.name,obj=v1)
                    self.data_flow.add_node(v2.name,obj=v2)
                    
    
                else:
                    v=point(name,[x,y,z])
                    v.alignment=alignment_v.label
                    v.notes=notes_v.value
                    self.vectors[v.name]=v
                    self.data_flow.add_node(v.name,obj=v)
                    
                self._sort()
                name_v.value=''
                notes_v.value=''
                print('Done')
        m1_button.on_click(m1_click)                
        #######################################################################
        #######################################################################
        
        #######################################################################
        # Creating method 2 data requirments.
        #######################################################################
        
        p1_l = widgets.HTML('<b>Reference Point 1')
        p1_v = widgets.Dropdown()
        p1_b = widgets.VBox([p1_l,p1_v])
        
        p2_l = widgets.HTML('<b>Reference Point 2')
        p2_v = widgets.Dropdown()
        p2_b = widgets.VBox([p2_l,p2_v])
        
        p1_v.options=p2_v.options=dict(self.points)
        
        m2_block = widgets.VBox([p1_b,p2_b,vectors_sub])
        
        m2_button = widgets.Button(description='Apply',icon='check')
        m2_button.layout=layout100px
        def m2_click(dummy):
            with vectors_sub:
                if name_v.value=='':
                    print('Please enter a valid name in the Vector Name field')
                    return
                
                if alignment_v.label in 'RL':
                    p1 = p1_v.value
                    p2 = p2_v.value
                    name1 = alignment_v.value+name_v.value
                    v1  = point(name1,p1-p2)
                    v1.alignment = alignment_v.label
                    v2 = v1.m_object
                    v2.name=v2.name.replace('hp','ov')
                    v2.alignment='RL'.replace(alignment_v.label,'')
                    
                    self.vectors[v1.name]=v1
                    self.vectors[v2.name]=v2
                    
                    self.data_flow.add_node(v1.name,obj=v1)
                    self.data_flow.add_edge(p1.name,v1.name,attr='p1')
                    self.data_flow.add_edge(p2.name,v1.name,attr='p2')
                    self.data_flow.add_node(v2.name,obj=v2)
                    self.data_flow.add_edge(p1.name,v2.name,attr='p1')
                    self.data_flow.add_edge(p2.name,v2.name,attr='p2')

                    
                elif alignment_v.label =='S':
                    p1 = p1_v.value
                    p2 = p2_v.value
                    name = alignment_v.value+name_v.value
                    v  = point(name,p1-p2)
                    v.alignment = alignment_v.label
                    self.vectors[name]=v
                    self.data_flow.add_node(v.name,obj=v)
                    self.data_flow.add_edge(p1.name,v.name,attr='p1')
                    self.data_flow.add_edge(p2.name,v.name,attr='p2')

                
                self._sort()
                name_v.value=''
                notes_v.value=''
                print('Done!')
        m2_button.on_click(m2_click)
        #######################################################################       
        #######################################################################
        
        #######################################################################
        # Defining the change behavior of methods dropdown
        #######################################################################
        def method_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                vectors_out.clear_output()
                vectors_sub.clear_output()
                p1_v.options=p2_v.options=dict(self.points)
                with vectors_out:
                    if change['new']=='User Entered Value':
                        block = widgets.VBox([m1_block,m1_button])
                        return ipy.display.display(block)
                    elif change['new']=='Relative Position':
                        block = widgets.VBox([m2_block,m2_button])
                        return ipy.display.display(block)
                    elif change['new']=='Normal to Plane':
                        return ipy.display.display(m1_block)
                
        method_v.observe(method_change)
        #######################################################################
        #######################################################################
        
        return widgets.VBox([main_data_block,vectors_out])

    
    
    def add_bodies(self):
        '''
        Creating the adding bodies gui
        '''
        
        #######################################################################
        # Creating the main outout window and its components
        #######################################################################
        main_out    = widgets.Output()


        name_l = widgets.HTML('<b>Body Name')
        name_v = widgets.Text(placeholder='Enter Body Name')
        name_b = widgets.VBox([name_l,name_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'rbr_','L':'rbl_','S':'rbs_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])
        
        notes_l = widgets.HTML('<b>Notes')
        notes_v = widgets.Textarea(placeholder='Brief description ...')
        notes_v.layout=widgets.Layout(width='300px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        body_type_l = widgets.HTML('<b>Body Type')
        body_type_v = widgets.ToggleButton(description='Float',value=False,icon='cube',layout=layout80px)
        body_type_b = widgets.VBox([body_type_l,body_type_v],layout=widgets.Layout(left='50px'))
        def type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                if body_type_v.value:
                    body_type_v.description=' Mount'
                    body_type_v.icon='plug'
                else:
                    body_type_v.description=' Float'
                    body_type_v.icon='cube'
        body_type_v.observe(type_change)
        
        align_type_box = widgets.HBox([alignment_b,body_type_b])
        
        create_default_button = widgets.Button(description=' New Body',icon='plus',tooltip='Create Body with default values')
        create_default_button.layout=layout120px
        def create_default_click(b):
            main_out.clear_output()
            with main_out:
                if name_v.value=='':
                    print('ERROR: Please Enter a Valid Name')
                    return
                
                if alignment_v.label in 'RL':
                    body_name_l = 'rbl_'+name_v.value
                    bod_l = rigid(body_name_l)
                    bod_l.notes=notes_v.value
                    
                    body_name_r = 'rbr_'+name_v.value
                    bod_r = rigid(body_name_r)
                    bod_r.notes=notes_v.value
                   
                    bod_l.alignment='L'
                    bod_r.alignment='R'
                    
                    self.bodies[body_name_l]=bod_l
                    self.bodies[body_name_r]=bod_r
                    
                    self.data_flow.add_node(bod_l.name,obj=bod_l)
                    self.data_flow.add_node(bod_r.name,obj=bod_r)

                
                elif alignment_v.label=='S':
                    body_name = 'rbs_'+name_v.value
                    bod = rigid(body_name)
                    bod.alignment='S'
                    self.bodies[body_name]=bod
                    self.data_flow.add_node(bod.name,obj=bod)
                
                self._sort()
                bodies_dropdown.options=dict(self.bodies)
        create_default_button.on_click(create_default_click)
        
        main_block_data = widgets.VBox([name_b,align_type_box,notes_b])
        main_block = widgets.VBox([main_block_data,separator50,create_default_button,main_out])

        #######################################################################
        #######################################################################
        #######################################################################
        
        
        #######################################################################
        # Creating the explicitly defined properties window and its components
        #######################################################################
        accord1_out = widgets.Output()
        
        bodies_dropdown_l = widgets.HTML('<b>Select Body')
        bodies_dropdown   = widgets.Dropdown(options=dict(self.bodies))
        bodies_dropdown_b = widgets.VBox([bodies_dropdown_l,bodies_dropdown])
        bodies_dropdown.layout=bodies_dropdown_l.layout=layout120px

        def bodies_dropdown_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                with accord1_out:
                    if bodies_dropdown.label==None:
                        return
                    else:
                        mass_v.value = bodies_dropdown.value.mass
                        alignment_v.label=bodies_dropdown.value.alignment
                        notes_v.value = bodies_dropdown.value.notes
                        x.value,y.value,z.value = bodies_dropdown.value.R
                        ixx.value,ixy.value,ixz.value,iyy.value,iyz.value,izz.value=[bodies_dropdown.value.J.flatten()[v] for v in[0,3,6,4,7,8]]
                        xx.value,xy.value,xz.value,yx.value,yy.value,yz.value,zx.value,zy.value,zz.value=bodies_dropdown.value.dcm.T.flatten()
        bodies_dropdown.observe(bodies_dropdown_change)

        mass_l = widgets.HTML('<b>Body Mass')
        mass_v = widgets.FloatText(value=1)
        mass_b = widgets.VBox([mass_l,mass_v])
        mass_v.layout=widgets.Layout(width='80px')
        
        reference_point_lable = widgets.HTML('<b>C.G Location')
        x_l = widgets.Label(value='$R_x$')
        y_l = widgets.Label(value='$R_y$')
        z_l = widgets.Label(value='$R_z$')
        x_l.layout=y_l.layout=z_l.layout=layout80px
    
        x = widgets.FloatText()
        y = widgets.FloatText()
        z = widgets.FloatText()
        x.layout=y.layout=z.layout=layout80px
        
        cg_lables_block = widgets.HBox([x_l,y_l,z_l])
        cg_input_block  = widgets.HBox([x,y,z])
        cg_block        = widgets.VBox([reference_point_lable,cg_lables_block,cg_input_block])
        
        ############################################################################
        # Inertia Moments Data
        ############################################################################
        inertia_lable = widgets.HTML('<b>Inertia Tensor')
        ixx = widgets.FloatText(value=1,layout=layout80px)
        iyy = widgets.FloatText(value=1,layout=layout80px)
        izz = widgets.FloatText(value=1,layout=layout80px)
        ixy = widgets.FloatText(layout=layout80px)
        ixz = widgets.FloatText(layout=layout80px)
        iyz = widgets.FloatText(layout=layout80px)
        iyx = widgets.FloatText(disabled=True,layout=layout80px)
        izx = widgets.FloatText(disabled=True,layout=layout80px)
        izy = widgets.FloatText(disabled=True,layout=layout80px)
        
        widgets.link((ixy, 'value'), (iyx, 'value'))
        widgets.link((ixz, 'value'), (izx, 'value'))
        widgets.link((iyz, 'value'), (izy, 'value'))
                                  
        
        ix = widgets.VBox([ixx,ixy,ixz])
        iy = widgets.VBox([iyx,iyy,iyz])
        iz = widgets.VBox([izx,izy,izz])
        
        inertia_tensor = widgets.HBox([ix,iy,iz])
        inertia_block  = widgets.VBox([inertia_lable,inertia_tensor])
        
        ############################################################################
        # Inertia Reference Frame Data
        ############################################################################
        frame_lable = widgets.HTML('<b>Inertia Frame')
        xx = widgets.FloatText(value=1,layout=layout80px)
        xy = widgets.FloatText(layout=layout80px)
        xz = widgets.FloatText(layout=layout80px)
        yx = widgets.FloatText(layout=layout80px)
        yy = widgets.FloatText(value=1,layout=layout80px)
        yz = widgets.FloatText(layout=layout80px)
        zx = widgets.FloatText(layout=layout80px)
        zy = widgets.FloatText(layout=layout80px)
        zz = widgets.FloatText(value=1,layout=layout80px)
        
        
        x_vector = widgets.VBox([xx,xy,xz])
        y_vector = widgets.VBox([yx,yy,yz])
        z_vector = widgets.VBox([zx,zy,zz])
        
        frame_matrix = widgets.HBox([x_vector,y_vector,z_vector])
        frame_block  = widgets.VBox([frame_lable,frame_matrix])

        
        add_inertia_button = widgets.Button(description=' Apply',icon='check',tooltip='Add inertia values')
        add_inertia_button.layout=layout80px
        def add_inertia_click(dummy):
            accord1_out.clear_output()
            with accord1_out:
                if name_v.value=='':
                    print('ERROR: Please Enter a Valid Name')
                    return
                body = bodies_dropdown.value
                
                mass      = mass_v.value
                iner_tens = np.array([[ixx.value,ixy.value,ixz.value],
                                      [ixy.value,iyy.value,iyz.value],
                                      [ixz.value,iyz.value,izz.value]])
    
                if body.alignment in 'RL':
                    name_1      = body.name
                    cm_1        = vector([x.value,y.value,z.value])
                    ref_frame_1 = np.array([[xx.value,yx.value,zx.value],
                                          [xy.value,yy.value,zy.value],
                                          [xz.value,yz.value,zz.value]])
                    body_1 = rigid(name_1,mass,iner_tens,cm_1,ref_frame_1)
                    
                    name_2      = body.m_name
                    cm_2        = vector([x.value,-y.value,z.value])
                    ref_frame_2 = np.array([[xx.value,yx.value,zx.value],
                                            [-xy.value,-yy.value,-zy.value],
                                            [xz.value,yz.value,zz.value]])
                    body_2 = rigid(name_2,mass,iner_tens,cm_2,ref_frame_2)
                   
                    body_2.alignment='RL'.replace(body.alignment,'')
                    
                    self.bodies[name_1]=body_1
                    self.bodies[name_2]=body_2
                    
                    self.data_flow.add_node(body_1.name,obj=body_1)
                    self.data_flow.add_node(body_2.name,obj=body_2)


                    
                    print('Bodies added : \n %s \n %s' %(name_1,name_2) )
                
                elif alignment_v.label=='S':
                    cm        = vector([x.value,y.value,z.value])
                    ref_frame = np.array([[xx.value,yx.value,zx.value],
                                          [xy.value,yy.value,zy.value],
                                          [xz.value,yz.value,zz.value]])
                    bod = rigid(body.name,mass,iner_tens,cm,ref_frame)
                    bod.alignment='S'
                    self.bodies[body.name]=bod
                    self.data_flow.add_node(bod.name,obj=bod)
                    
                    print('Body added : \n %s' %body.name )
                
                self._sort()
                bodies_dropdown.options=dict(self.bodies)
                
        add_inertia_button.on_click(add_inertia_click)

        data_block = widgets.VBox([bodies_dropdown_b,mass_b,cg_block,inertia_block,frame_block,separator50,add_inertia_button])
        window2_block = widgets.HBox([data_block,accord1_out])
               
        #######################################################################
        #######################################################################
        #######################################################################
        
        
        #######################################################################
        # Creating the explicitly defined properties window and its components
        #######################################################################
        
        accord2_out = widgets.Output()
        feed_back_2 = widgets.Output()

        geometries_dict={'':'','Cylinder':circular_cylinder}
                
        geo_name_l = widgets.HTML('<b>Geometry Name',layout=layout120px)
        geo_name_v = widgets.Text(placeholder='Enter Geometry Name',layout=layout120px)
        geo_name_b = widgets.VBox([geo_name_l,geo_name_v])
    
        
        geometries_l = widgets.HTML('<b>Select Geometry',layout=layout120px)
        geometries_v = widgets.Dropdown(options=geometries_dict,layout=layout120px)
        geometries_b = widgets.VBox([geometries_l,geometries_v])
    
        p1_l = widgets.HTML('<b>Point 1',layout=layout120px)
        p2_l = widgets.HTML('<b>Point 2',layout=layout120px)
        outer_l = widgets.HTML('<b>Outer Diameter',layout=layout120px)
        inner_l = widgets.HTML('<b>Inner Diameter',layout=layout120px)
        
        p1_v = widgets.Dropdown(options=dict(self.points),layout=layout120px)
        p2_v = widgets.Dropdown(options=dict(self.points),layout=layout120px)
        outer_v = widgets.FloatText(layout=layout120px)
        inner_v = widgets.FloatText(layout=layout120px)
        
        p1_b = widgets.VBox([p1_l,p1_v])
        p2_b = widgets.VBox([p2_l,p2_v])
        outer_b = widgets.VBox([outer_l,outer_v])
        inner_b = widgets.VBox([inner_l,inner_v])
        
        cylinder_window = widgets.VBox([p1_b,p2_b,outer_b,inner_b])
        
        
        # Creating apply button to assign selected geometry to assigned body
        assign_geometry = widgets.Button(description='Apply',tooltip='Assign Geometry to Body',layout=layout120px)
        def assign_click(b):
            with accord2_out:
                if bodies_dropdown.value==None:
                    print('ERROR: Please Chose a Body Object')
                    return 
                body = bodies_dropdown.value
                if body.alignment in 'RL':
                    body_1 = bodies_dropdown.value
                    body_2 = self.bodies[body_1.m_name]
                    
                    p1_1 = self.points[p1_v.label]
                    p1_2 = self.points[p1_v.value.m_name]
                    p2_1 = self.points[p2_v.label]
                    p2_2 = self.points[p2_v.value.m_name]
                    
                    geo_name_1 = body.name+'_'+geo_name_v.value
                    geo_name_2 = body.m_name+'_'+geo_name_v.value
                    
                    geo_1 = geometries_dict[geometries_v.label](geo_name_1,body_1,p1_1,p2_1,outer_v.value,inner_v.value)
                    geo_2 = geometries_dict[geometries_v.label](geo_name_2,body_2,p1_2,p2_2,outer_v.value,inner_v.value)
                    
                    self.geometries[geo_name_1] = geo_1
                    self.geometries[geo_name_2] = geo_2
                    
                    body_1.update_inertia()
                    body_2.update_inertia()
                    
                    self.data_flow.add_node(geo_name_1,obj=geo_1)
                    self.data_flow.add_node(geo_name_2,obj=geo_2)
                    
                    self.data_flow.add_edge(p1_1.name,geo_name_1,attr='p1')
                    self.data_flow.add_edge(p2_1.name,geo_name_1,attr='p2')
                    
                    self.data_flow.add_edge(geo_name_1,body.name)
                
                elif  body.alignment=='S':
                    p1 = self.pointsp1_v.value
                    p2 = self.pointsp2_v.value

                    geo_name = body.name+'_'+geo_name_v.value
                    
                    geo = geometries_dict[geometries_v.label](geo_name,body,p1,p2,outer_v.value,inner_v.value)
                    self.geometries[geo_name] = geo
                    body.update_inertia()
                    
                    self.data_flow.add_node(geo_name,obj=geo)
                    self.data_flow.add_edge(p1.name,geo_name)
                    self.data_flow.add_edge(p2.name,geo_name)
                    
                    self.data_flow.add_edge(geo_name,body.name)

                    
                geo_name_v.value=''
            
            feed_back_2.clear_output()
            with feed_back_2:
                print('Operation done at %s'%time.strftime('%I:%M:%S %p'))
                            
        assign_geometry.on_click(assign_click)
        
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                accord2_out.clear_output()
                with accord2_out:
                    p1_v.options=p2_v.options=dict(self.points)
                    if change['new']==geometries_dict['Cylinder']:
                        ipy.display.display(widgets.VBox([cylinder_window,assign_geometry]))
                    
        geometries_v.observe(on_change)
                
        
        geometries_define_inputs = widgets.VBox([bodies_dropdown_b,geo_name_b,geometries_b,accord2_out,feed_back_2])
        
                
        main_block2 = widgets.Accordion()
        main_block2.children=[window2_block,geometries_define_inputs]
        main_block2.set_title(0,'EXPLICTLY DEFINE BODY PROPERTIES')
        main_block2.set_title(1,'DEFINE BODY GEOMETRY')
        main_block2.selected_index=1
        
        return widgets.VBox([main_block,main_block2])
    
    
    
    
    def add_joints(self):
        
        main_out = widgets.Output()
        logger_out = widgets.Output()
        
        refresh_button = widgets.Button(description=' Refresh',icon='undo')
        refresh_button.layout=layout100px
        def refresh_click(dummy):
            with main_out:
                body_i_v.options=body_j_v.options=dict(self.bodies)
                axis1_v.options=axis2_v.options=dict(self.vectors)
        refresh_button.on_click(refresh_click)

        joints_dict = {'Spherical': spherical,
                       'Revolute' : revolute,
                       'Translational': translational,
                       'Cylinderical':cylindrical,
                       'Universal': universal}
        
        joint_type_l = widgets.HTML('<b>Joint Type')
        joint_type_v = widgets.Dropdown(options=joints_dict)
        joint_type_b = widgets.VBox([joint_type_l,joint_type_v])
        
        joint_inputs_out = widgets.Output()
        
        name_l = widgets.HTML('<b>Joint Name',layout=layout120px)
        name_v = widgets.Text(placeholder='joint name',layout=layout120px)
        name_b = widgets.VBox([name_l,name_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'jcr_','L':'jcl_','S':'jcs_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])
        
        notes_l = widgets.HTML('<b>Notes')
        notes_v = widgets.Textarea(placeholder='Brief description ...')
        notes_v.layout=widgets.Layout(width='300px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        location_l = widgets.HTML('<b>Locaction',layout=layout120px)
        location_v = widgets.Dropdown(options=dict(self.points),layout=layout120px)
        location_b = widgets.VBox([location_l,location_v])
        
        body_i_l = widgets.HTML('<b>Body i',layout=layout120px)
        body_i_v = widgets.Dropdown(layout=layout120px)
        body_i_v.options=dict(self.bodies)
        body_i_b = widgets.VBox([body_i_l,body_i_v])
        
        body_j_l = widgets.HTML('<b>Body j')
        body_j_v = widgets.Dropdown(layout=layout120px)
        body_j_v.options=dict(self.bodies)
        body_j_b = widgets.VBox([body_j_l,body_j_v])
        
        axis1_l = widgets.HTML('<b>Axis 1')
        axis1_v = widgets.Dropdown(options=dict(self.vectors))
        axis1_b = widgets.VBox([axis1_l,axis1_v])
        
        axis2_l = widgets.HTML('<b>Axis 2')
        axis2_v = widgets.Dropdown(options=dict(self.vectors))
        axis2_b = widgets.VBox([axis2_l,axis2_v])
        
        field1 = widgets.VBox([refresh_button,separator50,name_b,alignment_b,notes_b,separator50,joint_type_b])
        field2 = widgets.HBox([location_b,body_i_b,body_j_b])
        
        uni_field = widgets.VBox([axis1_b,axis2_b])
        
        def joint_type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                joint_inputs_out.clear_output()
                with joint_inputs_out:
                    axis1_v.options=axis2_v.options=dict(self.vectors)
                    location_v.options=dict(self.points)
                    if change['new']==joints_dict['Universal']:
                        ipy.display.display(uni_field)
                    elif change['new']==joints_dict['Spherical'] :
                        joint_inputs_out.clear_output()
                    else:
                        ipy.display.display(axis1_b)
        joint_type_v.observe(joint_type_change)
                        
        creat_joint_button = widgets.Button(description='Apply',icon='check')
        creat_joint_button.layout=layout100px
        def creat_joint_click(dummy):
            with joint_inputs_out:
                
                if alignment_v.label in "RL":
                    name_1 = alignment_v.value+name_v.value
                    alignment_1 = alignment_v.label
                    alignment_2 = 'RL'.replace(alignment_v.label,'')
                    
                    loc_1 = location_v.value
                    loc_2 = self.points[loc_1.m_name]
                    
                    bodyi_1 = body_i_v.value
                    bodyi_2 = self.bodies[bodyi_1.m_name]
                    
                    bodyj_1 = body_j_v.value
                    bodyj_2 = self.bodies[bodyj_1.m_name]
                    

                    if joint_type_v.label=='Universal':
                        axis1_1 = axis1_v.value
                        axis1_2 = self.vectors[axis1_1.m_name.replace('hp','ov')]
                        
                        axis2_1 = axis2_v.value
                        axis2_2 = self.vectors[axis2_1.m_name.replace('hp','ov')]
                        
                        
                        j1=joint_type_v.value(name_1,loc_1,bodyi_1,bodyj_1,axis1_1,axis2_1)
                        j1.alignment=alignment_1
                        j2=joint_type_v.value(j1.m_name,loc_2,bodyi_2,bodyj_2,axis1_2,axis2_2)
                        j2.alignment=alignment_2
                        
                        self.data_flow.add_edge(axis2_1.name,j1.name)
                        self.data_flow.add_edge(axis1_1.name,j1.name)

                    
                    elif joint_type_v.label=='Spherical' :
                        
                        j1=joint_type_v.value(name_1,loc_1,bodyi_1,bodyj_1)
                        j1.alignment=alignment_1
                        j2=joint_type_v.value(j1.m_name,loc_2,bodyi_2,bodyj_2)
                        j2.alignment=alignment_2
                    
                    
                    else:
                        axis1_1 = axis1_v.value
                        axis1_2 = self.vectors[axis1_1.m_name.replace('hp','ov')]
                        j1=joint_type_v.value(name_1,loc_1,bodyi_1,bodyj_1,axis1_1)
                        j1.alignment=alignment_1
                        j2=joint_type_v.value(j1.m_name,loc_2,bodyi_2,bodyj_2,axis1_2)
                        j2.alignment=alignment_2
                        
                        self.data_flow.add_edge(axis1_1.name,j1.name)

                    
                    j1.notes=j2.notes=notes_v.value
                    
                    self.joints[j1.name]=j1
                    self.joints[j2.name]=j2
                    
                    self.data_flow.add_node(j1.name,obj=j1)
                    self.data_flow.add_edge(loc_1.name,j1.name,attr='location')
                    self.data_flow.add_edge(bodyi_1.name,j1.name,attr='i_body')
                    self.data_flow.add_edge(bodyj_1.name,j1.name,attr='j_body')
                
                elif alignment_v.label=='S':
                    joint_name = alignment_v.value+name_v.value
                    loc        = location_v.value
                    bodyi      = body_i_v.value
                    bodyj      = body_j_v.value
                    notes      = notes_v.value
                    
                    if joint_type_v.label=='Universal':
                        j=joint_type_v.value(joint_name,loc,bodyi,bodyj,axis1_v.value,axis2_v.value)
                        self.data_flow.add_edge(axis2_v.label,j1.name)
                        self.data_flow.add_edge(axis1_v.label,j1.name)
                        
                    elif joint_type_v.label=='Spherical' :
                        j=joint_type_v.value(joint_name,loc,bodyi,bodyj)
                    else:
                        j=joint_type_v.value(joint_name,loc,bodyi,bodyj,axis1_v.value)
                        self.data_flow.add_edge(axis1_v.label,j1.name)
                    
                    j.notes=notes
                    self.joints[joint_name]=j
                    
                    self.data_flow.add_node(j.name,obj=j)
                    self.data_flow.add_edge(loc.name,j.name)
                    self.data_flow.add_edge(bodyi.name,j.name)
                    self.data_flow.add_edge(bodyj.name,j.name)


                
            with logger_out:
                print('Joint "%s" Created at %s'%(name_v.value,time.strftime('%I:%M:%S %p')))
        creat_joint_button.on_click(creat_joint_click)
        
        out_block = widgets.HBox([widgets.VBox([field1,field2,joint_inputs_out,creat_joint_button]),logger_out])
        
        return out_block
    
    
    
    
    def add_forces(self):
        
        #######################################################################
        # Creating the gui for defining force elements.
        #######################################################################
        
        force_elements_out = widgets.Output()
        
        name_l = widgets.HTML('<b>Force Element Name',layout=layout120px)
        name_v = widgets.Text(placeholder='force element name',layout=layout200px)
        name_b = widgets.VBox([name_l,name_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'fer_','L':'fel_','S':'fes_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])
        
        notes_l = widgets.HTML('<b>Notes')
        notes_v = widgets.Textarea(placeholder='Brief description ...')
        notes_v.layout=widgets.Layout(width='300px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        common_data_block = widgets.VBox([name_b,alignment_b,notes_b])
        #######################################################################

        pi_l = widgets.HTML('<b>Locaction on body i',layout=layout120px)
        pi_v = widgets.Dropdown(options=dict(self.points),layout=layout120px)
        pi_b = widgets.VBox([pi_l,pi_v])
        
        pj_l = widgets.HTML('<b>Locaction on body j',layout=layout120px)
        pj_v = widgets.Dropdown(options=dict(self.points),layout=layout120px)
        pj_b = widgets.VBox([pj_l,pj_v])
        
        body_i_l = widgets.HTML('<b>Body i',layout=layout120px)
        body_i_v = widgets.Dropdown(layout=layout120px)
        body_i_v.options=dict(self.bodies)
        body_i_b = widgets.VBox([body_i_l,body_i_v])
        
        body_j_l = widgets.HTML('<b>Body j')
        body_j_v = widgets.Dropdown(layout=layout120px)
        body_j_v.options=dict(self.bodies)
        body_j_b = widgets.VBox([body_j_l,body_j_v])
        
        strut_data_block  = widgets.VBox([pi_b,pj_b,body_i_b,body_j_b])

        
        #######################################################################
        ######################### Stiffness Data GUI ##########################
        #######################################################################
        stiffness_label = widgets.HTML('<b><u>Stiffness Data')

        stiffness_out        = widgets.Output()
        stiffness_tabel_out  = widgets.Output()
        stiffness_plot_out   = widgets.Output()

        stiffness_df = widgets.ValueWidget()
        stiffness_df.value = pd.DataFrame([[0,0],],columns=['Force','Deflection'])
        stiffness_qg = qgrid.QGridWidget(df=stiffness_df.value,show_toolbar=True)
        stiffness_qg.layout=widgets.Layout(width='400px')
        
        with stiffness_tabel_out:
            ipy.display.display(stiffness_qg)
        
        
        stiffness_apply_changes = widgets.Button(description='Apply',icon='check',tooltip='Apply Changes',layout=layout100px)
        def stiffness_apply_click(dummy):
            with stiffness_tabel_out:
                stiffness_df.value=stiffness_qg.get_changed_df()
                print('Changes Applied at %s'%time.strftime('%I:%M:%S %p'))
        stiffness_apply_changes.on_click(stiffness_apply_click)
        
        stiffness_export_button =  widgets.Button(description='Export',icon='upload',tooltip='Export to excel sheet',layout=layout100px)
        def stiffness_export_click(dummy):
            with stiffness_tabel_out:
                stiffness_qg.get_changed_df().to_excel(name_v.value+'_stiffness_data.xlsx')
                print('Data Exported as "%s"  at %s'%(name_v.value+'_stiffness_data.xlsx',time.strftime('%I:%M:%S %p')))
        stiffness_export_button.on_click(stiffness_export_click)
        
        stiffness_import_button =  widgets.Button(description='Import',icon='download',tooltip='Import to excel sheet',layout=layout100px)
        def stiffness_import_click(dummy):
            with stiffness_tabel_out:
                stiffness_df.value=pd.read_excel(name_v.value+'_stiffness_data.xlsx')
                stiffness_qg.df=stiffness_df.value
                print('Data Imported from "%s"  at %s'%(name_v.value+'_stiffness_data.xlsx',time.strftime('%I:%M:%S %p')))
        stiffness_import_button.on_click(stiffness_import_click)
        
        
        stiffness_toggle = widgets.ToggleButton(value=False,description=' Show/Edit',icon='edit',tooltip='Show and edit data',layout=layout100px)
        def stiffness_toggle_click(change):
            if change['type'] == 'change' and change['name'] == 'value':
                if stiffness_toggle.value:
                    with stiffness_out:
                        ipy.display.display(stiffness_visual_data)
                else:
                    stiffness_out.clear_output()
        stiffness_toggle.observe(stiffness_toggle_click)

        stiffness_buttons = widgets.HBox([stiffness_apply_changes,stiffness_export_button,stiffness_import_button])
        stiffness_visual_data = widgets.VBox([stiffness_tabel_out,stiffness_buttons,stiffness_plot_out])
        
        stiffness_visual_toggle = widgets.VBox([stiffness_label,stiffness_toggle,stiffness_out])
        #######################################################################
        #######################################################################

        #######################################################################
        ########################## Damping Data GUI ###########################
        #######################################################################
        damping_label = widgets.HTML('<b><u>Damping Data')
        
        damping_out        = widgets.Output()
        damping_tabel_out  = widgets.Output()
        damping_plot_out   = widgets.Output()

        damping_df = widgets.ValueWidget()
        damping_df.value = pd.DataFrame([[0,0],],columns=['Force','Velocity'])
        
        damping_qg = qgrid.QGridWidget(df=damping_df.value,show_toolbar=True)
        damping_qg.layout=widgets.Layout(width='400px')
        
        with damping_tabel_out:
            ipy.display.display(damping_qg)
        
        
        damping_apply_changes = widgets.Button(description='Apply',icon='check',tooltip='Apply Changes',layout=layout100px)
        def damping_apply_click(dummy):
            with damping_tabel_out:
                damping_df.value=damping_qg.get_changed_df()
                print('Changes Applied at %s'%time.strftime('%I:%M:%S %p'))
        damping_apply_changes.on_click(damping_apply_click)
        
        damping_export_button =  widgets.Button(description='Export',icon='upload',tooltip='Export to excel sheet',layout=layout100px)
        def damping_export_click(dummy):
            with damping_tabel_out:
                damping_qg.get_changed_df().to_excel(name_v.value+'_damping_data.xlsx')
                print('Data Exported as "%s"  at %s'%(name_v.value+'_damping_data.xlsx',time.strftime('%I:%M:%S %p')))
        damping_export_button.on_click(damping_export_click)
        
        damping_import_button =  widgets.Button(description='Import',icon='download',tooltip='Import to excel sheet',layout=layout100px)
        def damping_import_click(dummy):
            with damping_tabel_out:
                damping_df.value=pd.read_excel(name_v.value+'_damping_data.xlsx')
                damping_qg.df=damping_df.value
                print('Data Imported as "%s"  at %s'%(name_v.value+'_damping_data.xlsx',time.strftime('%I:%M:%S %p')))
        damping_import_button.on_click(damping_import_click)
        
        
        damping_toggle = widgets.ToggleButton(value=False,description=' Show/Edit',icon='edit',tooltip='Show and edit data',layout=layout100px)
        def damping_toggle_click(change):
            if change['type'] == 'change' and change['name'] == 'value':
                if damping_toggle.value:
                    with damping_out:
                        ipy.display.display(damping_visual_data)
                else:
                    damping_out.clear_output()
        damping_toggle.observe(damping_toggle_click)

        damping_buttons = widgets.HBox([damping_apply_changes,damping_export_button,damping_import_button])
        damping_visual_data = widgets.VBox([damping_tabel_out,damping_buttons,damping_plot_out])
        
        damping_visual_toggle = widgets.VBox([damping_label,damping_toggle,damping_out])
        
        #######################################################################
        #######################################################################
        rh_stroke_l = widgets.HTML('<b> Stroke at Ride Hieght',layout=layout200px)
        rh_stroke_v = widgets.FloatText(layout=layout120px)
        rh_stroke_b = widgets.VBox([rh_stroke_l,rh_stroke_v])
        
        
        
        #######################################################################
        ###################### Adding Force Element ###########################
        #######################################################################
        
        add_force_element_button = widgets.Button(description='Add',icon='check',tooltip='Add Force Element',layout=layout100px)
        def add_force_element_click(dummy):
            with force_elements_out:
                if name_v.value=='':
                    print('ERROR: Please Enter a Valid Name!')
                    return
                name = name_v.value
                stiffness = stiffness_df.value
                damping   = damping_df.value
                rh_stroke = rh_stroke_v.value
                if alignment_v.label in 'RL':
                    body_i_1 = body_i_v.value
                    body_i_2 = self.bodies[body_i_1.m_name]
                    
                    body_j_1 = body_j_v.value
                    body_j_2 = self.bodies[body_j_1.m_name]
                    
                    pi_1 = self.points[pi_v.label]
                    pi_2 = self.points[pi_v.value.m_name]
                    pj_1 = self.points[pj_v.label]
                    pj_2 = self.points[pj_v.value.m_name]
                    
                    name_1 = alignment_v.value+name
                    strut_1 = air_strut(name_1,pi_1,body_i_1,pj_1,body_j_1,stiffness,damping,rh_stroke)
                    strut_1.alignment = alignment_v.label
                    
                    name_2 = strut_1.m_name
                    strut_2 = air_strut(name_2,pi_2,body_i_2,pj_2,body_j_2,stiffness,damping,rh_stroke)
                    strut_2.alignment = 'RL'.replace(alignment_v.label,'')
                    
                    self.forces[strut_1.name]=strut_1
                    self.forces[strut_2.name]=strut_2
                    
                    self.data_flow.add_node(strut_1.name,obj=strut_1)
                    self.data_flow.add_edge(pi_1.name,strut_1.name)
                    self.data_flow.add_edge(pj_1.name,strut_1.name)
                    self.data_flow.add_edge(body_i_1.name,strut_1.name)
                    self.data_flow.add_edge(body_j_1.name,strut_1.name)

                    
                    
                elif  alignment_v.label=='S':
                    name = alignment_v.value+name
                    pi = pi_v.value
                    pj = pj_v.value
                    bodyi = body_i_v.value
                    bodyj = body_j_v.value
                    
                    strut = air_strut(name,pi,bodyi,pj,bodyj,stiffness,damping,rh_stroke)
                    self.forces[strut.name]=strut
                    
                    self.data_flow.add_node(strut.name,obj=strut)
                    self.data_flow.add_edge(pi.name,strut.name)
                    self.data_flow.add_edge(pj.name,strut.name)
                    self.data_flow.add_edge(bodyi.name,strut.name)
                    self.data_flow.add_edge(bodyj.name,strut.name)

                
                print('Done')
        add_force_element_button.on_click(add_force_element_click)
        
        #######################################################################
        #######################################################################

        strut_data_tab    = widgets.VBox([strut_data_block,rh_stroke_b,
                                          separator100,
                                          stiffness_visual_toggle,
                                          separator100,
                                          damping_visual_toggle,
                                          separator100,
                                          add_force_element_button,force_elements_out])
                            
        
        return widgets.VBox([common_data_block,separator100,strut_data_tab])
        
    
    
    
    def add_actuators(self):
        
        actuator_elements_out = widgets.Output()
        
        name_l = widgets.HTML('<b>Actuator Name',layout=layout120px)
        name_v = widgets.Text(placeholder='actuator name',layout=layout200px)
        name_b = widgets.VBox([name_l,name_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'mcr_','L':'mcl_','S':'mcs_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])
        
        notes_l = widgets.HTML('<b>Notes')
        notes_v = widgets.Textarea(placeholder='Brief description ...')
        notes_v.layout=widgets.Layout(width='300px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        common_data_block = widgets.VBox([name_b,alignment_b,notes_b])
        #######################################################################
        
        actuator_type_l = widgets.HTML('<b>Actuator Type')
        actuator_type_v = widgets.Dropdown(options={'':'','RotationalDrive':rotational_drive,'AbsoluteLocating':absolute_locating})
        actuator_type_b = widgets.VBox([actuator_type_l,actuator_type_v])
        
        #######################################################################
        #################### Rotational Drive Actuator ########################
        #######################################################################
        
        
        
        
        
        
        
        #######################################################################
        #######################################################################
        def type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                return
        
        return widgets.VBox([common_data_block,separator100,actuator_type_b,separator50])
    
    
    def parallel_travel(self):
        parallel_out = widgets.Output()
        
        name_l = widgets.HTML('<b>Simulation Name',layout=layout120px)
        name_v = widgets.Text(placeholder='name',layout=layout120px)
        name_b = widgets.HBox([name_l,name_v])
        
        notes_l = widgets.HTML('<b>Notes',layout=layout200px)
        notes_v = widgets.Textarea(placeholder='Brief description ...')
        notes_v.layout=widgets.Layout(width='300px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        jounce_l = widgets.HTML('<b>Jounce Travel',layout=layout120px)
        jounce_v = widgets.FloatText(layout=layout100px)
        jounce_b = widgets.HBox([jounce_l,jounce_v])
        
        rebound_l = widgets.HTML('<b>Rebound Travel',layout=layout120px)
        rebound_v = widgets.FloatText(layout=layout100px)
        rebound_b = widgets.HBox([rebound_l,rebound_v])
        
        timesteps_l = widgets.HTML('<b>Simulation Steps',layout=layout120px)
        timesteps_v = widgets.FloatText(layout=layout100px)
        timesteps_b = widgets.HBox([timesteps_l,timesteps_v])
        
        
        
        wheel_hub_left=self.joints['jcl_hub_bearing']
        wheel_hub_right=self.joints['jcr_hub_bearing']
        
        wheel_drive_left     = rotational_drive('mcl_rotational_lock',wheel_hub_left)
        wheel_drive_right    = rotational_drive('mcr_rotational_lock',wheel_hub_right)
        
        vertical_travel_left = absolute_locating('mcl_vertical',self.bodies['rbl_wheel_hub'],'z')
        vertical_travel_right = absolute_locating('mcr_vertical',self.bodies['rbr_wheel_hub'],'z')
                
        
        
        names = [i.name for i in [wheel_drive_left,wheel_drive_right,vertical_travel_left,vertical_travel_right]]
        
        self.actuators = pd.Series([wheel_drive_left,wheel_drive_right,vertical_travel_left,vertical_travel_right],index=names)

        topology_writer(self.bodies,self.joints,self.actuators,self.forces,'ST100_datafile_kinematic')

        run_button = widgets.Button(description='Run',icon='play',tooltip='Add Force Element',layout=layout100px)
        def run_click(dummy):
            with parallel_out:
                total_travel = jounce_v.value+rebound_v.value
                v_shift = (0.5*total_travel) - rebound_v.value
                t=np.linspace(0,2*np.pi,timesteps_v.value)
                wheel_drive_left.pos_array=np.zeros((len(t),))
                wheel_drive_right.pos_array=np.zeros((len(t),))
                
                vertical_motion=0.5*total_travel*np.sin(t-np.arcsin(v_shift/(0.5*total_travel)))+v_shift
                
                vertical_travel_left.pos_array=600+vertical_motion
                vertical_travel_right.pos_array=600+vertical_motion
                self.soln=kds(self.bodies,self.joints,self.actuators,'ST100_datafile_kinematic',t)
                print('Done!!')
        run_button.on_click(run_click)
        
        
        
        common_data_block = widgets.VBox([name_b,separator100,jounce_b,rebound_b,timesteps_b,separator100,notes_b,separator100,run_button,parallel_out])
        return common_data_block
    
    
    
    
    def data_processing(self):
        
        plots_out = widgets.Output()
        
        data_type_l = widgets.HTML('<b> Select Data',layout=layout120px)
        data_type_v = widgets.Select(options={'Position':0,'Velocity':1,'Acceleration':2},layout=layout120px)
        data_type_b = widgets.VBox([data_type_l,data_type_v])
        
        in_object_selector_l = widgets.HTML('<b> Select Object',layout=layout120px)
        in_object_selector_v = widgets.Select(options=self.bodies.index,layout=layout200px)
        in_object_selector_b = widgets.VBox([in_object_selector_l,in_object_selector_v])
        
        in_attribute_selector_l = widgets.HTML('<b> Select Attribute',layout=layout120px)
        in_attribute_selector_v = widgets.Select(options={'x':'.x','y':'.y','z':'.z'},layout=layout120px)
        in_attribute_selector_b = widgets.VBox([in_attribute_selector_l,in_attribute_selector_v])
        
        independent_l = widgets.HTML('<b><u>Independent Variable')
        dependent_l   = widgets.HTML('<b><u>Dependent Variable')
        
        de_object_selector_l = widgets.HTML('<b> Select Object',layout=layout120px)
        de_object_selector_v = widgets.Select(options=self.bodies.index,layout=layout200px)
        de_object_selector_b = widgets.VBox([de_object_selector_l,de_object_selector_v])
        
        de_attribute_selector_l = widgets.HTML('<b> Select Attribute',layout=layout120px)
        de_attribute_selector_v = widgets.Select(options={'x':'.x','y':'.y','z':'.z'},layout=layout120px)
        de_attribute_selector_b = widgets.VBox([de_attribute_selector_l,de_attribute_selector_v])
        
        
        show_button = widgets.Button(description='Show',icon='image',tooltip='Show Plot',layout=layout100px)
        def show_click(dummy):
            plots_out.clear_output()
            with plots_out:
                index_ind = in_object_selector_v.value+in_attribute_selector_v.value
                index_dep = de_object_selector_v.value+de_attribute_selector_v.value
                indpendent_data  = self.soln[data_type_v.value][index_ind]
                dependent_data   = self.soln[data_type_v.value][index_dep]
                
                plt.figure('l',figsize=(10,4))
                plt.title(index_ind+' vs '+index_dep,color='white')
                plt.plot(indpendent_data,dependent_data,label=index_dep)
                plt.legend()
                plt.tick_params(axis='x', colors='white')
                plt.tick_params(axis='y', colors='white')
                plt.grid()
                plt.show()
        show_button.on_click(show_click)
        
        selectors = widgets.VBox([data_type_b,separator100,
                                  independent_l,
                                  widgets.HBox([in_object_selector_b,in_attribute_selector_b]),
                                  separator100,
                                  dependent_l,
                                  widgets.HBox([de_object_selector_b,de_attribute_selector_b]),
                                  separator100,
                                  show_button])
        
        
        return widgets.VBox([selectors,plots_out])
        
    
    
    def show_data_flow(self):
        out = widgets.Output()
        with out:
            plt.figure('Model Data Flow',figsize=(8,5))
            nx.draw_circular(self.data_flow,with_labels=True)
            plt.show()
        return out
    
    def object_successors(self,object_node):
        out = widgets.Output()
        with out:
            plt.figure('Object Dependencies',figsize=(8,5))
            dependencies = nx.DiGraph(nx.edge_dfs(self.data_flow,object_node))
            nx.draw_circular(nx.DiGraph(dependencies),with_labels=True)
            plt.show()
        return out
    
    def object_predecessors(self,object_node):
        out = widgets.Output()
        with out:
            plt.figure('Object Dependencies',figsize=(8,5))
            dependencies = nx.DiGraph([(i[0],i[1]) for i in nx.edge_dfs(self.data_flow,object_node,'reverse')])
            nx.draw_circular(nx.DiGraph(dependencies),with_labels=True)
            plt.show()
        return out
    
    def edit_node(self,node):
        g=self.data_flow
        
        for e in nx.edge_dfs(g,node):
            try:
                g.node[e[1]]['obj'].__setattr__(g.edges[e]['attr'],g.node[e[0]]['obj'])
                print('Editing object "%s" and updating attribute "%s" in dependency "%s" \n' %(e[0],g.edges[e]['attr'],e[1]))
            except KeyError:
                print('Not Found \n')
                pass

    
    def show(self):
        
        ipy.display.display(ipy.display.Markdown('## VEHICLE DYNAMICS MODELING AND SIMULATION TOOL'))
        
        buttons = widgets.HBox([self.new_model(),self.open_model(),self.save_model(),self.save_model_copy()])
        out = widgets.VBox([buttons,self.out])
        return out
        
        
