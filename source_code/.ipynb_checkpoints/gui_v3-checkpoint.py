# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:57:14 2018

@author: khaled.ghobashy
"""


import ipywidgets as widgets
import IPython as ipy
import qgrid
import pandas as pd
from base import point, vector
from bodies_inertia import rigid
from inertia_properties import circular_cylinder
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
        self.model      = pd.Series()
        
        self.name = ''
        
        self.points_dataframe = pd.DataFrame(columns=['x','y','z','Alignment','Notes'])
        self.bodies_dataframe = pd.DataFrame(columns=['mass','Rx','Ry','Rz',
                                                      'Ixx','Iyx','Izx',
                                                      'Ixy','Iyy','Izy',
                                                      'Ixz','Iyz','Izz',
                                                      'xx','yx','zx',
                                                      'xy','yy','zy',
                                                      'xz','yz','zz'])

    
    
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
        
                        
    def save_model(self):
        save_out = widgets.Output()
        
        save_button = widgets.Button(description='Save Model',tooltip='Save Model Binary files')
        def save_click(dummy):
            with save_out:
                f=savefile_dialog()
                if f=='':
                    return
                self.model['points']=self.points.sort_index()
                self.model['bodies']=self.bodies.sort_index()
                self.model['joints']=self.joints.sort_index()
                self.model['geometries']=self.geometries.sort_index()
                self.model['vectors']=self.vectors.sort_index()
                
                self.model.to_pickle(f)
                print('Done!')
        save_button.on_click(save_click)
        
        return widgets.VBox([save_button,save_out])
                
    def open_model(self):
        open_out = widgets.Output()
        
        open_button = widgets.Button(description='Open Model',tooltip='Open Model Binary files')
        def open_click(dummy):
            open_out.clear_output()
            with open_out:
                f=openfile_dialog()
                if f=='':
                    return
                self.model=pd.read_pickle(f)
                self.points,self.bodies,self.joints,self.geometries=self.model
                for i in self.points:
                    self.points_dataframe.loc[i.name]=[i.x,i.y,i.z,i.alignment,i.notes]
                
                self._sort()
                
                fields = widgets.Accordion()
                fields.children=[self.add_point(),self.add_vectors(),self.add_bodies(),self.add_joints()]
                fields.set_title(0,'SYSTEM POINTS')
                fields.set_title(1,'SYSTEM MARKERS')
                fields.set_title(2,'SYSTEM BODIES')
                fields.set_title(3,'SYSTEM JOINTS')
                print('Done!')
                return ipy.display.display(fields)
                
        open_button.on_click(open_click)
        
        
        return widgets.VBox([open_button,open_out])
    
    def new_model(self):
        new_out = widgets.Output()
        
        new_button = widgets.Button(description='New Model',tooltip='Creat a New Model')
        def new_click(dummy):
            with new_out:
                fields = widgets.Accordion()
                fields.children=[self.add_point(),self.add_vectors(),self.add_bodies(),self.add_joints()]
                fields.set_title(0,'SYSTEM POINTS')
                fields.set_title(1,'SYSTEM MARKERS')
                fields.set_title(2,'SYSTEM BODIES')
                fields.set_title(3,'SYSTEM JOINTS')
                return ipy.display.display(fields)
                
        new_button.on_click(new_click)
        
        return widgets.VBox([new_button,new_out])
        
    def add_point(self):
        
        tabs = widgets.Tab()
        tabs.set_title(0,'ADD NEW POINT')
        tabs.set_title(1,'POINTS TABLE')
        tabs.set_title(2,'IMPORT / EXPORT')
        
        tab1_out = widgets.Output()
        tab2_out = widgets.Output()
        
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
        notes_v = widgets.Textarea(placeholder='Optional brief note/describtion.')
        notes_v.layout=widgets.Layout(width='350px',height='55px')
        notes_b = widgets.VBox([notes_l,notes_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'hpr_','L':'hpl_','S':'hps_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])        
        edit_l = widgets.HTML('<b>Edit Point')
        
        points_dropdown = widgets.Dropdown()
        points_dropdown.options = self._filter_points()
        points_dropdown.layout=layout120px
        
        
        field1 = widgets.HBox([name_b,x_b,y_b,z_b])
        field2 = widgets.VBox([alignment_b,notes_b])
        field3 = widgets.VBox([edit_l,points_dropdown])
        
        
        
        
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
                    self.points[p1.name]=p1
                    self.points[p2.name]=p2
                    
                    self.points_dataframe.loc[p1.name]=[x,p1.y,z,p1.alignment,notes_v.value]
                    self.points_dataframe.loc[p2.name]=[x,p2.y,z,p2.alignment,notes_v.value]
    
                else:
                    p=point(name,[x,y,z])
                    p.alignment=alignment_v.label
                    p.notes=notes_v.value
                    self.points[p.name]=p
                    self.points_dataframe.loc[p.name]=[x,y,z,p.alignment,notes_v.value]
                    
                
                name_v.value=''
                notes_v.value=''
                self._sort()
                points_dropdown.options=dict(self.points)
                
            with tab2_out:
                ipy.display.display(qgrid.QgridWidget(df=self.points_dataframe))
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
        
        
        
        tab1_content = widgets.VBox([field1,field2,add_button,separator100,field3,tab1_out])
        tab2_content = widgets.VBox([qgrid.QgridWidget(df=self.points_dataframe)],layout=widgets.Layout(width='550px'))
        
        tabs.children=[tab1_content,tab2_content]
        
        return tabs
    
    
    def add_vectors(self):

        vectors_out = widgets.Output()
        
        vector_name_l = widgets.HTML('<b>Vector Name')
        vector_name_v = widgets.Text(placeholder='vector name here')
        vector_name_b = widgets.VBox([vector_name_l,vector_name_v])
        
        alignment_l = widgets.HTML('<b>Alignment')
        alignment_v = widgets.ToggleButtons(options={'R':'ovr_','L':'ovl_','S':'ovs_'})
        alignment_v.style.button_width='40px'
        alignment_b = widgets.VBox([alignment_l,alignment_v])

        
        p1_l = widgets.HTML('<b>Reference Point 1')
        p1_v = widgets.Dropdown()
        p1_b = widgets.VBox([p1_l,p1_v])
        
        p2_l = widgets.HTML('<b>Reference Point 2')
        p2_v = widgets.Dropdown()
        p2_b = widgets.VBox([p2_l,p2_v])
        
        p1_v.options=p2_v.options=dict(self.points)
        
        add_vector_button = widgets.Button(description='Apply',tooltip='add vector')
        def add_vector_click(dummy):
            with vectors_out:
                if alignment_v.label in 'RL':
                    p1 = p1_v.value
                    p2 = p2_v.value
                    name1 = alignment_v.value+vector_name_v.value
                    v1  = p1-p2
                    v1.alignment = alignment_v.label
                    name2 = v1.mirrored+vector_name_v.value
                    v2 = p1-p2
                    v2.y*-1
                    v2.alignment='RL'.replace(alignment_v.label,'')
                    
                    self.vectors[name1]=v1
                    self.vectors[name2]=v2
                elif alignment_v.label =='S':
                    p1 = p1_v.value
                    p2 = p2_v.value
                    name = alignment_v.value+vector_name_v.value
                    v  = p1-p2
                    v.alignment = alignment_v.label
                    self.vectors[name]=v
                print('Done!')
        add_vector_button.on_click(add_vector_click)
                
        
        vectors_field = widgets.VBox([separator100,vector_name_b,alignment_b,p1_b,p2_b,add_vector_button,vectors_out])
        vectors_tab = widgets.Tab([vectors_field])
        vectors_tab.set_title(0,'Defining Vectors')
        
        return vectors_field

    
    
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
        
        
        create_default_button = widgets.Button(description=' New Body',icon='plus',tooltip='Create Body with default values')
        create_default_button.layout=layout120px
        def create_default_click(b):
            main_out.clear_output()
            with main_out:
                if name_v.value=='':
                    print('ERROR: Please Enter a Valid Name')
                    return
                
                if alignment_v.label in ['R','L']:
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
                elif alignment_v.label=='S':
                    body_name = 'rbs_'+name_v.value
                    bod = rigid(body_name)
                    bod.alignment='S'
                    self.bodies[body_name]=bod
                
                bodies_dropdown.options=self._filter_bodies()
        create_default_button.on_click(create_default_click)
        
        main_block_data = widgets.VBox([name_b,alignment_b,notes_b])
        main_block = widgets.VBox([main_block_data,separator50,create_default_button,main_out])

        #######################################################################
        #######################################################################
        #######################################################################
        
        
        #######################################################################
        # Creating the explicitly defined properties window and its components
        #######################################################################
        accord1_out = widgets.Output()
        
        bodies_dropdown_l = widgets.HTML('<b>Select Body')
        bodies_dropdown   = widgets.Dropdown(options=self._filter_bodies())
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
            with accord1_out:
                if name_v.value=='':
                    print('ERROR: Please Enter a Valid Name')
                    return
                
                mass      = mass_v.value
                iner_tens = np.array([[ixx.value,ixy.value,ixz.value],
                                      [ixy.value,iyy.value,iyz.value],
                                      [ixz.value,iyz.value,izz.value]])
    
                if alignment_v.label in ['R','L']:
                    body_name_l = 'rbl_'+name_v.value
                    cm_l        = vector([x.value,-abs(y.value),z.value])
                    ref_frame_l = np.array([[xx.value,yx.value,zx.value],
                                          [xy.value,yy.value,zy.value],
                                          [xz.value,yz.value,zz.value]])
                    bod_l = rigid(body_name_l,mass,iner_tens,cm_l,ref_frame_l)
                    
                    body_name_r = 'rbr_'+name_v.value
                    cm_r        = vector([x.value,abs(y.value),z.value])
                    ref_frame_r = np.array([[xx.value,yx.value,zx.value],
                                            [-xy.value,-yy.value,-zy.value],
                                            [xz.value,yz.value,zz.value]])
                    bod_r = rigid(body_name_r,mass,iner_tens,cm_r,ref_frame_r)
                   
                    bod_l.alignment='L'
                    bod_r.alignment='R'
                    
                    self.bodies[body_name_l]=bod_l
                    self.bodies[body_name_r]=bod_r
                elif alignment_v.label=='S':
                    body_name = 'rbs_'+name_v.value
                    cm        = vector([x.value,y.value,z.value])
                    ref_frame = np.array([[xx.value,yx.value,zx.value],
                                          [xy.value,yy.value,zy.value],
                                          [xz.value,yz.value,zz.value]])
                    bod = rigid(body_name,mass,iner_tens,cm,ref_frame)
                    bod.alignment='S'
                    self.bodies[body_name]=bod
                
                bodies_dropdown.options=self._filter_bodies()
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
                    body_2 = self.bodies[body_1.mirrored]
                    
                    p1_1 = self.points[p1_v.label]
                    p1_2 = self.points[p1_v.value.mirrored]
                    p2_1 = self.points[p2_v.label]
                    p2_2 = self.points[p2_v.value.mirrored]
                    
                    geo_name_1 = body.name+'_'+geo_name_v.value
                    geo_name_2 = body.mirrored+'_'+geo_name_v.value
                    
                    self.geometries[geo_name_1]=geometries_dict[geometries_v.label](geo_name_1,body_1,p1_1,p2_1,outer_v.value,inner_v.value)
                    self.geometries[geo_name_2]=geometries_dict[geometries_v.label](geo_name_2,body_2,p1_2,p2_2,outer_v.value,inner_v.value)
                    
                    body_1.update_inertia()
                    body_2.update_inertia()
                
                elif  body.alignment=='S':
                    p1 = self.pointsp1_v.value
                    p2 = self.pointsp2_v.value

                    geo_name = body.name+'_'+geo_name_v.value
                    self.geometries[geo_name]=geometries_dict[geometries_v.label](geo_name,body,p1,p2,outer_v.value,inner_v.value)
                    body.update_inertia()
                    
                geo_name_v.value=''
                            
        assign_geometry.on_click(assign_click)
        
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                accord2_out.clear_output()
                with accord2_out:
                    p1_v.options=p2_v.options=dict(self.points)
                    if change['new']==geometries_dict['Cylinder']:
                        ipy.display.display(widgets.VBox([cylinder_window,assign_geometry]))
                    
        geometries_v.observe(on_change)
                
        
        geometries_define_inputs = widgets.VBox([bodies_dropdown_b,geo_name_b,geometries_b,accord2_out])
        
                
        main_block2 = widgets.Accordion()
        main_block2.children=[window2_block,geometries_define_inputs]
        main_block2.set_title(0,'EXPLICTLY DEFINE BODY PROPERTIES')
        main_block2.set_title(1,'DEFINE BODY GEOMETRY')
        main_block2.selected_index=1
        
        return widgets.VBox([main_block,main_block2])
    
    
    
    
    def add_joints(self):
        
        main_out = widgets.Output()
        
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
                    loc_1 = self.points[location_v.label]
                    loc_2 = self.points[loc_1.mirrored]
                    
                    bodyi_1 = body_i_v.value
                    bodyi_2 = self.bodies[bodyi_1.mirrored]
                    
                    bodyj_1 = body_j_v.value
                    bodyj_2 = self.bodies[bodyj_1.mirrored]
                    

                    if joint_type_v.label=='Universal':
                        axis1_1 = self.vectors[axis1_v.label]
                        axis1_2 = self.vectors[axis1_1.mirrored+axis1_v.label[4:]]
                        
                        axis2_1 = self.vectors[axis2_v.label]
                        axis2_2 = self.vectors[axis2_1.mirrored+axis2_v.label[4:]]
                        
                        j1=joint_type_v.value(loc_1,bodyi_1,bodyj_1,axis1_1,axis2_1)
                        j2=joint_type_v.value(loc_2,bodyi_2,bodyj_2,axis1_2,axis2_2)
                    
                    elif joint_type_v.label=='Spherical' :
                        j1=joint_type_v.value(loc_1,bodyi_1,bodyj_1)
                        j2=joint_type_v.value(loc_2,bodyi_2,bodyj_2)
                    else:
                        axis1_1 = self.vectors[axis1_v.label]
                        axis1_2 = self.vectors[axis1_1.mirrored+axis1_v.label[4:]]
                        j1=joint_type_v.value(loc_1,bodyi_1,bodyj_1,axis1_1)
                        j2=joint_type_v.value(loc_2,bodyi_2,bodyj_2,axis1_2)
                    
                    j1.notes=j2.notes=notes_v.value
                    
                    joint_name_1 = alignment_v.value+name_v.value
                    joint_name_2 = j1.mirror+name_v.value
                    self.joints[joint_name_1]=j1
                    self.joints[joint_name_2]=j2
                
                elif alignment_v.label=='S':
                    joint_name = alignment_v.value+name_v.value
                    loc        = location_v.value
                    bodyi      = body_i_v.value
                    bodyj      = body_j_v.value
                    notes      = notes_v.value
                    
                    if joint_type_v.label=='Universal':
                        j=joint_type_v.value(loc,bodyi,bodyj,axis1_v.value,axis2_v.value)
                    elif joint_type_v.label=='Spherical' :
                        j=joint_type_v.value(loc,bodyi,bodyj)
                    else:
                        j=joint_type_v.value(loc,bodyi,bodyj,axis1_v.value)
                    
                    j.notes=notes
                    self.joints[joint_name]=j

                
                
                print('Done!')
        creat_joint_button.on_click(creat_joint_click)
        
        
        
        return widgets.VBox([field1,field2,joint_inputs_out,creat_joint_button])
    
    
    
    def show(self):
               
        
        return widgets.VBox([self.new_model(),self.save_model(),self.open_model()])
        
        


















