

import ipywidgets as widgets
import IPython as ipy
import qgrid
import pandas as pd
from base import point, vector
from bodies_inertia import rigid
from inertia_properties import composite_geometry,circular_cylinder
import numpy as np

def adding_points_gui(existing_csv_file=None):
    
    # Creating lables widgets for inputs
    # ###############################################################################
    name_lable = widgets.Label(value='$Name$')
    x_lable = widgets.Label(value='$x$')
    y_lable = widgets.Label(value='$y$')
    z_lable = widgets.Label(value='$z$')

    coordinates_lables = widgets.HBox([name_lable,x_lable,y_lable,z_lable])
    name_lable.layout=widgets.Layout(width='130px')
    x_lable.layout=y_lable.layout=z_lable.layout=widgets.Layout(width='100px')
    # ###############################################################################
    
    #################################################################################
    # Creating input fields widgets for inputs
    #################################################################################
    name = widgets.Text(placeholder='Point Name')
    x = widgets.FloatText()
    y = widgets.FloatText()
    z = widgets.FloatText()

    name.layout=widgets.Layout(width='120px')
    x.layout=y.layout=z.layout=widgets.Layout(width='100px')

    inputs = widgets.HBox([name,x,y,z])

    notes = widgets.Textarea(placeholder='Optional brief note/describtion.')
    notes.layout=widgets.Layout(width='350px',height='55px')
    alignment=widgets.RadioButtons(options={'R':'hpr_','L':'hpl_','S':'hps_'})
    #################################################################################


    apply_button = widgets.Button(description='Apply')
    existing_points = widgets.Dropdown()
    import_point = widgets.Button(description='Import')
    export_button= widgets.Button(description='Export',tooltip='export to .xls')
    import_button= widgets.Button(description='Import',tooltip='import from .xls')
    out   = widgets.Output()
    out2  = widgets.Output()
    out_3 = widgets.Output()
    
    export_name = widgets.Text(placeholder='Enter File Name')
    import_name = widgets.Text(placeholder='Enter File Name')

    if existing_csv_file==None:
        data =pd.DataFrame(columns=['x','y','z','Alignment','Notes'])
        points_objects = pd.Series()
    else:
        data = pd.read_csv(existing_csv_file)
        points_objects = pd.Series()
    
    def export_points(b):
        out_3.clear_output()
        with out_3:
            data.to_excel(export_name.value+'.xls')
            ipy.display.display('Export Done!')
    
    def import_points(b):
        out_3.clear_output()
        out2.clear_output()
        with out_3:
            data_import=pd.read_excel(import_name.value+'.xls')
            for i in data_import.index:
                data.loc[i]=data_import.loc[i]
            print('Import Done!')
        with out2:
            ipy.display.display(qgrid.QgridWidget(df=data))
        
        with out:
            for i in data.index:
                p=point(i,data.loc[i]['x':'z'])
                p.alignment=i[0:4]
                points_objects[i]=p
            existing_points.options=dict(zip([name[4:] for name in points_objects.index ],points_objects.index))
            
        
    def add(b):
        out.clear_output()
        out2.clear_output()
        #name.value=('UN-NAMED' if name.value=='' else name.value)
        with out:
            if name.value=='':
                print('Please enter a valid name in the Point Name field')
                return 
            
            if alignment.value!='hps_':
                nl='hpl_'+name.value
                pl=point(nl,[x.value,-abs(y.value),z.value])
                pl.alignment='hpl_'
                points_objects[nl]=pl
                
                nr='hpr_'+name.value
                pr=point(nr,[x.value,abs(y.value),z.value])
                pr.alignment='hpr_'
                points_objects[nr]=pr
                
                data.loc[nl]=[x.value,-abs(y.value),z.value,'L',notes.value]
                data.loc[nr]=[x.value, abs(y.value),z.value,'R',notes.value]

            else:
                n=alignment.value+name.value
                p=point(n,[x.value,y.value,z.value])
                p.alignment=alignment.value
                points_objects[n]=p
                data.loc[n]=[x.value,y.value,z.value,'S',notes.value]
                
            
            name.value=''
            existing_points.options=dict(zip([name[4:] for name in points_objects.index ],points_objects.index))
            
        with out2:
            ipy.display.display(qgrid.QgridWidget(df=data))
            #tab.children=[content,ww]
            

    def edit(b):
        out.clear_output()
        with out:
            name.value=existing_points.value[4:]
            x.value=points_objects[existing_points.value].x
            y.value=points_objects[existing_points.value].y
            z.value=points_objects[existing_points.value].z
            alignment.value=points_objects[existing_points.value].alignment



    apply_button.on_click(add)
    import_point.on_click(edit)
    export_button.on_click(export_points)
    import_button.on_click(import_points)

    import_area = widgets.HBox([import_name,import_button])
    export_area = widgets.HBox([export_name,export_button])
    
    edit_lable = widgets.Label(value='$Edit$ $Point$')
    edit_window=widgets.VBox([edit_lable,widgets.HBox([existing_points,import_point])])
    #edit_window.layout=widgets.Layout(top='15px',bottom='10px')

    content = widgets.VBox([coordinates_lables,inputs,widgets.HBox([notes,alignment]),apply_button,widgets.VBox(),out,edit_window])
    children = [content,out2,widgets.VBox([export_area,import_area,out_3])]
    tab = widgets.Tab()
    tab.children = children
    tab.set_title(0,'ADD NEW POINT')
    tab.set_title(1,'POINTS TABLE')
    tab.set_title(2,'IMPORT / EXPORT')

    return tab,points_objects,data




def add_bodies_gui(points=[]):
    
    main_out = widgets.Output()

    # Defining used gui blocks
    bodies_objects = pd.Series()

    body_name_lable = widgets.Label(value='$Body$ $Name$')
    body_name_value = widgets.Text(placeholder='Enter Body Name')
    body_name_block = widgets.VBox([body_name_lable,body_name_value])

    add_body_default   = widgets.Button(description='Apply',tooltip='Create Body with default values')
    def create_body_default(b):
        with main_out:
            if body_name_value.value=='':
                print('ERROR: Please Enter a Valid Name')
                return
            body_name = body_name_value.value
            bod = rigid(body_name)
            bodies_objects[body_name]=bod
            exsisting_bodies_value.options=bodies_objects
            body_name_value.value=''
    
    add_body_default.on_click(create_body_default)
    # accordion 1 data
    ############################################################################
    # CG Data
    ############################################################################
    mass_label = widgets.Label(value='$Body$ $Mass$')
    mass_value = widgets.FloatText(value=1)
    mass_block = widgets.VBox([mass_label,mass_value])
    mass_value.layout=widgets.Layout(width='80px')
    
    reference_point_lable = widgets.Label(value='$C.G$ $Location$')
    x_lable = widgets.Label(value='$R_x$')
    y_lable = widgets.Label(value='$R_y$')
    z_lable = widgets.Label(value='$R_z$')
    x_lable.layout=y_lable.layout=z_lable.layout=widgets.Layout(width='80px')

    x = widgets.FloatText()
    y = widgets.FloatText()
    z = widgets.FloatText()
    x.layout=y.layout=z.layout=widgets.Layout(width='80px')
    
    cg_lables_block = widgets.HBox([x_lable,y_lable,z_lable])
    cg_input_block  = widgets.HBox([x,y,z])
    cg_block        = widgets.VBox([reference_point_lable,cg_lables_block,cg_input_block])
    
    ############################################################################
    # Inertia Moments Data
    ############################################################################
    layout_80px = widgets.Layout(width='80px')
    inertia_lable = widgets.Label(value='$Inertia$ $Tensor$')
    ixx = widgets.FloatText(value=1,layout=layout_80px)
    iyy = widgets.FloatText(value=1,layout=layout_80px)
    izz = widgets.FloatText(value=1,layout=layout_80px)
    ixy = widgets.FloatText(layout=layout_80px)
    ixz = widgets.FloatText(layout=layout_80px)
    iyz = widgets.FloatText(layout=layout_80px)
    iyx = widgets.FloatText(disabled=True,layout=layout_80px)
    izx = widgets.FloatText(disabled=True,layout=layout_80px)
    izy = widgets.FloatText(disabled=True,layout=layout_80px)
    
    widgets.link((ixy, 'value'), (iyx, 'value'))
    widgets.link((ixz, 'value'), (izx, 'value'))
    widgets.link((iyz, 'value'), (izy, 'value'))
                              
    
    ix = widgets.VBox([ixx,ixy,ixz])
    iy = widgets.VBox([iyx,iyy,iyz])
    iz = widgets.VBox([izx,izy,izz])
    
    inertia_tensor = widgets.HBox([ix,iy,iz])
    inertia_block  = widgets.VBox([inertia_lable,inertia_tensor],layout=widgets.Layout(top='40px'))
    
    ############################################################################
    # Inertia Reference Frame Data
    ############################################################################
    frame_lable = widgets.Label(value='$Inertia$ $Frame$')
    xx = widgets.FloatText(value=1,layout=layout_80px)
    xy = widgets.FloatText(layout=layout_80px)
    xz = widgets.FloatText(layout=layout_80px)
    yx = widgets.FloatText(layout=layout_80px)
    yy = widgets.FloatText(value=1,layout=layout_80px)
    yz = widgets.FloatText(layout=layout_80px)
    zx = widgets.FloatText(layout=layout_80px)
    zy = widgets.FloatText(layout=layout_80px)
    zz = widgets.FloatText(value=1,layout=layout_80px)
    
    
    x_vector = widgets.VBox([xx,xy,xz])
    y_vector = widgets.VBox([yx,yy,yz])
    z_vector = widgets.VBox([zx,zy,zz])
    
    reference_frame    = widgets.HBox([x_vector,y_vector,z_vector])
    inertia_ref_block  = widgets.VBox([frame_lable,reference_frame],layout=widgets.Layout(top='60px'))
    ############################################################################
    
    ############################################################################
    # Defining Interactive Buttons for adding bodies
    ############################################################################
    accord1_out = widgets.Output()
    add_body    = widgets.Button(description='Apply',tooltip='submitt selected data')
    
    def create_body(b):
        with accord1_out:
            body_name = body_name_value.value
            mass      = mass_value.value
            cm        = vector([x.value,y.value,z.value])
            ref_frame = np.array([[xx.value,yx.value,zx.value],
                                  [xy.value,yy.value,zy.value],
                                  [xz.value,yz.value,zz.value]])
            
            iner_tens = np.array([[ixx.value,ixy.value,ixz.value],
                                  [ixy.value,iyy.value,iyz.value],
                                  [ixz.value,iyz.value,izz.value]])
            
            bod = rigid(body_name,mass,iner_tens,cm,ref_frame)
            bodies_objects[body_name]=bod
            exsisting_bodies_value.options=bodies_objects
    
    add_body.on_click(create_body)
    
    body_data_block   = widgets.VBox([mass_block,cg_block,inertia_block,inertia_ref_block])
    body_data_block.layout=widgets.Layout(width='300px',height='500px')
    accordion_1_block = widgets.HBox([body_data_block,add_body])
    ############################################################################
    
    
    
    ############################################################################
    # accordion 2 data - Creating Geometries
    ############################################################################
    geometries_objects = pd.Series()
    accord2_out        = widgets.Output()
    
    geometries_dict={'':'','Cylinder':circular_cylinder}
    
    exsisting_bodies_label = widgets.Label(value='$Select$ $Body$')
    exsisting_bodies_value = widgets.Dropdown(options=bodies_objects)
    exsisting_bodies_block = widgets.VBox([exsisting_bodies_label,exsisting_bodies_value])
    
    geo_name_label = widgets.Label(value='$Geometry$ $Name$')
    geo_name_value = widgets.Text(placeholder='Enter Geometry Name')
    geo_name_block = widgets.VBox([geo_name_label,geo_name_value])

    
    geometries_label = widgets.Label(value='$Select$ $Geometry$')
    geometries_value = widgets.Dropdown(options=geometries_dict)
    geometries_block = widgets.VBox([geometries_label,geometries_value])
    layout_120px     = widgets.Layout(width='120px')

    p1_label = widgets.Label(value='$Point$ $1$',layout=layout_120px)
    p2_label = widgets.Label(value='$Point$ $2$',layout=layout_120px)
    outer_label = widgets.Label(value='$Outer$ $Diameter$',layout=layout_120px)
    inner_label = widgets.Label(value='$Inner$ $Diameter$',layout=layout_120px)
    
    p1_value = widgets.Dropdown(options=[name for name in points.index],layout=layout_120px)
    p2_value = widgets.Dropdown(options=[name for name in points.index],layout=layout_120px)
    outer_value = widgets.FloatText(layout=layout_120px)
    inner_value = widgets.FloatText(layout=layout_120px)
    
    p1_block = widgets.VBox([p1_label,p1_value])
    p2_block = widgets.VBox([p2_label,p2_value])
    outer_block = widgets.VBox([outer_label,outer_value])
    inner_block = widgets.VBox([inner_label,inner_value])
    
    cylinder_window = widgets.VBox([p1_block,p2_block,outer_block,inner_block])
    
    
    # Creating apply button to assign selected geometry to assigned body
    assign_geometry = widgets.Button(description='Apply',tooltip='Assign Geometry to Body')
    def assign_click(b):
        with accord2_out:
            body = exsisting_bodies_value.value
            geo_name = body.name+'_'+geo_name_value.value
            geometries_objects[geo_name]=geometries_dict[geometries_value.label](geo_name,body,points[p1_value.value],points[p2_value.value],outer_value.value,inner_value.value)
            body.update_inertia()
            geo_name_value.value=''
                        
    assign_geometry.on_click(assign_click)
    
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            accord2_out.clear_output()
            with accord2_out:
                if change['new']==geometries_dict['Cylinder']:
                    ipy.display.display(cylinder_window)
                
    geometries_value.observe(on_change)
    
    accordion_2_block = widgets.VBox([exsisting_bodies_block,geo_name_block,geometries_block,accord2_out,assign_geometry])
    
    
    
    
    
    accord = widgets.Accordion()
    accord_children = [accordion_1_block,accordion_2_block]
    accord.children = accord_children
    accord.set_title(0,'EXPLICTLY DEFINE BODY PROPERTIES')
    accord.set_title(1,'DEFINE BODY GEOMETRY')
    accord.set_title(2,'IMPORT / EXPORT')
    accord.selected_index=1
    
    tab=widgets.Tab([widgets.VBox([body_name_block,add_body_default,accord,main_out]),])
    tab.set_title(0,'DEFINING NEW BODY')
    
    return tab,bodies_objects,geometries_objects



















