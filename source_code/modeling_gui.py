# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:45:37 2018

@author: khaled.ghobashy
"""
import ipywidgets as widgets
import IPython as ipy
import qgrid
import pandas as pd
from base import point, vector
from bodies_inertia import rigid
from inertia_properties import composite_geometry,circular_cylinder
import numpy as np

layout80px = widgets.Layout(width='80px')
layout100px = widgets.Layout(width='100px')
layout120px = widgets.Layout(width='120px')
layout200px = widgets.Layout(width='200px')

separator      = widgets.Label(value=''.join(55*['__']))
vertical_space = widgets.Label(value='\n')



class model(object):
    
    def __init__(self):
        self.tab       = widgets.Tab()
        
        self.points     = pd.Series()
        self.bodies     = pd.Series()
        self.joints     = pd.Series()
        self.geometries = pd.Series()
        
        self.points_dataframe = pd.DataFrame(columns=['x','y','z','Alignment','Notes'])
        self.points_dropdown_options = pd.Series()

         
    
    def add_point(self):
        
        tabs = widgets.Tab()
        tabs.set_title(0,'ADD NEW POINT')
        tabs.set_title(1,'POINTS TABLE')
        tabs.set_title(2,'IMPORT / EXPORT')
        
        tab1_out = widgets.Output()
        tab2_out = widgets.Output()
        tab3_out = widgets.Output()
        
        name_l = widgets.Label(value='$Name$')
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
        
                
        notes = widgets.Textarea(placeholder='Optional brief note/describtion.')
        notes.layout=widgets.Layout(width='350px',height='55px')
        alignment=widgets.RadioButtons(options={'R':'hpr_','L':'hpl_','S':'hps_'})
        
        edit_l = widgets.Label(value='$Edit$ $Point$')
        
        points_dropdown = widgets.Dropdown()
        points_dropdown.options = dict(zip([name[4:] for name in self.points.index ],self.points.index))
        points_dropdown.layout=layout120px
        
        
        field1 = widgets.HBox([name_b,x_b,y_b,z_b])
        field2 = widgets.HBox([notes,alignment])
        field3 = widgets.VBox([edit_l,points_dropdown])
        
        
        
        
        add_button = widgets.Button(description='Apply')
        def add_click(dummy):
            tab1_out.clear_output()
            tab2_out.clear_output()
            with tab1_out:
                name,x,y,z = [i.value for i in [name_v,x_v,y_v,z_v]]
                
                if name=='':
                    print('Please enter a valid name in the Point Name field')
                    return 
                
                if alignment.value!='hps_':
                    nl='hpl_'+name
                    pl=point(nl,[x,-abs(y),z])
                    pl.alignment='hpl_'
                    self.points[nl]=pl
                    
                    nr='hpr_'+name
                    pr=point(nr,[x,abs(y),z])
                    pr.alignment='hpr_'
                    self.points[nr]=pr
                    
                    self.points_dataframe.loc[nl]=[x,-abs(y),z,'L',notes.value]
                    self.points_dataframe.loc[nr]=[x, abs(y),z,'R',notes.value]
    
                else:
                    n='hps_'+name
                    p=point(n,[x,y,z])
                    p.alignment=alignment.value
                    self.points[n]=p
                    self.points_dataframe.loc[n]=[x,y,z,'S',notes.value]
                    
                
                name_v.value=''
                notes.value=''
                points_dropdown.options=dict(zip([name[4:] for name in self.points.index ],self.points.index))
                
            with tab2_out:
                ipy.display.display(qgrid.QgridWidget(df=self.points_dataframe))
        add_button.on_click(add_click)
        
        
        
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                with tab1_out:
                    name_v.value=points_dropdown.value[4:]
                    x_v.value=self.points[points_dropdown.value].x
                    y_v.value=self.points[points_dropdown.value].y
                    z_v.value=self.points[points_dropdown.value].z
                    alignment.value=self.points[points_dropdown.value].alignment
                    notes.value=self.points_dataframe.loc[points_dropdown.value]['Notes']
        points_dropdown.observe(on_change)
        
        
        
        export_l = widgets.Label(value='$Exporting$ $Points$')
        export_v = widgets.Text(placeholder='write file name here')
        export_v.layout=layout200px
        export_button = widgets.Button(description='Export')
        def export_click(b):
            tab3_out.clear_output()
            with tab3_out:
                self.points_dataframe.to_excel(export_v.value+'.xls')
                ipy.display.display('Export Done!')
        export_button.on_click(export_click)
        
        
        import_l = widgets.Label(value='$Importing$ $Points$')
        import_v = widgets.Text(placeholder='write file name here')
        import_v.layout=layout200px
        import_button = widgets.Button(description='Import')
        def import_click(b):
            tab3_out.clear_output()
            tab2_out.clear_output()
            with tab3_out:
                data_import=pd.read_excel(import_v.value+'.xls')
                for i in data_import.index:
                    self.points_dataframe.loc[i]=data_import.loc[i]
                print('Import Done!')
            with tab2_out:
                ipy.display.display(qgrid.QgridWidget(df=self.points_dataframe))
            
            with tab1_out:
                for i in self.points_dataframe.index:
                    p=point(i,self.points_dataframe.loc[i]['x':'z'])
                    p.alignment=i[0:4]
                    self.points[i]=p
                self.points_dropdown_options = dict(zip([name[4:] for name in  self.points.index ], self.points.index))
                points_dropdown.options = self.points_dropdown_options
        import_button.on_click(import_click)
        
        
        t3_f1 = widgets.HBox([import_v,import_button])
        t3_f2 = widgets.HBox([export_v,export_button])
        
        tab1_content = widgets.VBox([field1,field2,add_button,separator,field3])
        tab2_content = tab2_out
        tab3_content = widgets.VBox([import_l,t3_f1,separator,export_l,t3_f2,tab3_out])
        
        tabs.children=[tab1_content,tab2_content,tab3_content]
        
        return tabs
    
    
    def add_bodies(self):
        
        bodies_out  = widgets.Output()
        accord2_out = widgets.Output()


        name_l = widgets.Label(value='$Body$ $Name$')
        name_v = widgets.Text(placeholder='Enter Body Name')
        name_b = widgets.VBox([name_l,name_v])
        
        bodies_dropdown = widgets.Dropdown()
        bodies_dropdown.options = self.bodies
        
        add_button = widgets.Button(description='Apply',tooltip='Create Body with default values')
        def add_click(b):
            with bodies_out:
                if name_v.value=='':
                    print('ERROR: Please Enter a Valid Name')
                    return
                body_name = name_v.value
                bod = rigid(body_name)
                self.bodies[body_name]=bod
                bodies_dropdown.options=self.bodies
                name_v.value=''
        add_button.on_click(add_click)


        main_block = widgets.VBox([name_b,add_button])
    
        geometries_dict={'':'','Cylinder':circular_cylinder}
        
        bodies_dropdown_l = widgets.Label(value='$Select$ $Body$')
        bodies_dropdown_b = widgets.VBox([bodies_dropdown_l,bodies_dropdown])
        
        geo_name_l = widgets.Label(value='$Geometry$ $Name$')
        geo_name_v = widgets.Text(placeholder='Enter Geometry Name')
        geo_name_b = widgets.VBox([geo_name_l,geo_name_v])
    
        
        geometries_l = widgets.Label(value='$Select$ $Geometry$')
        geometries_v = widgets.Dropdown(options=geometries_dict)
        geometries_b = widgets.VBox([geometries_l,geometries_v])
    
        p1_l = widgets.Label(value='$Point$ $1$',layout=layout120px)
        p2_l = widgets.Label(value='$Point$ $2$',layout=layout120px)
        outer_l = widgets.Label(value='$Outer$ $Diameter$',layout=layout120px)
        inner_l = widgets.Label(value='$Inner$ $Diameter$',layout=layout120px)
        
        p1_v = widgets.Dropdown(options=self.points_dropdown_options,layout=layout120px)
        p2_v = widgets.Dropdown(options=self.points_dropdown_options,layout=layout120px)
        outer_v = widgets.FloatText(layout=layout120px)
        inner_v = widgets.FloatText(layout=layout120px)
        
        p1_b = widgets.VBox([p1_l,p1_v])
        p2_b = widgets.VBox([p2_l,p2_v])
        outer_b = widgets.VBox([outer_l,outer_v])
        inner_b = widgets.VBox([inner_l,inner_v])
        
        cylinder_window = widgets.VBox([p1_b,p2_b,outer_b,inner_b])
        
        
        # Creating apply button to assign selected geometry to assigned body
        assign_geometry = widgets.Button(description='Apply',tooltip='Assign Geometry to Body')
        def assign_click(b):
            with accord2_out:
                body = bodies_dropdown.value
                geo_name = body.name+'_'+geo_name_v.value
                self.geometries[geo_name]=geometries_dict[geometries_v.label](geo_name,body,self.points[p1_v.value],self.points[p2_v.value],outer_v.value,inner_v.value)
                body.update_inertia()
                geo_name_v.value=''
                            
        assign_geometry.on_click(assign_click)
        
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                accord2_out.clear_output()
                with accord2_out:
                    p1_v.options=p2_v.options=self.points_dropdown_options
                    if change['new']==geometries_dict['Cylinder']:
                        ipy.display.display(cylinder_window)
                    
        geometries_v.observe(on_change)
        
        accordion_2_block = widgets.VBox([bodies_dropdown_b,geo_name_b,geometries_b,accord2_out,assign_geometry])
        
        return widgets.VBox([main_block,accordion_2_block])
    
    
    def show(self):
        
        fields = widgets.Accordion()
        fields.children=[self.add_point(),self.add_bodies()]
        fields.set_title(0,'SYSTEM POINTS')
        fields.set_title(1,'SYSTEM BODIES')
        return fields
        
        


















