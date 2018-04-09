import ipywidgets as widgets
import qgrid
import pandas as pd
from base import *

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
    export_button= widgets.Button(description='Export csv file')
    out = widgets.Output()

    if existing_csv_file==None:
        data = pd.DataFrame({
                            'Name' : 'Point',
                            'x'    : np.array([0]*1),
                            'y'    : np.array([0]*1),
                            'z'    : np.array([0]*1),
                            'Alignment':'',
                            'Notes':''})
        data =pd.DataFrame(columns=['Name','x','y','z','Al','N'])
        points_objects = pd.Series()
    else:
        data = pd.read_csv(existing_csv_file)
        points_objects = pd.Series()
    
    ww=qgrid.QgridWidget(df=data)
    '''def export_points(b):
        out.clear_output()
        with out:
    '''        
        
    def add(b):
        out.clear_output()
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
                
                data.loc[len(data)]=[nl,x.value,-abs(y.value),z.value,'L',notes.value]
                data.loc[len(data)]=[nr,x.value, abs(y.value),z.value,'R',notes.value]

            else:
                n=alignment.value+name.value
                p=point(n,[x.value,y.value,z.value])
                p.alignment=alignment.value
                points_objects[n]=p
                data.loc[len(data)]=[n,x.value,y.value,z.value,'S',notes.value]
                
            
            name.value=''
            existing_points.options=[name for name in points_objects.index]
            ww=qgrid.QgridWidget(df=data)
            tab.children=[content,ww]
            

    def edit(b):
        out.clear_output()
        with out:
            name.value=existing_points.value
            x.value=points_objects[name.value].x
            y.value=points_objects[name.value].y
            z.value=points_objects[name.value].z
            alignment.value=points_objects[name.value].alignment



    apply_button.on_click(add)
    import_point.on_click(edit)

    edit_lable = widgets.Label(value='$Edit$ $Point$')
    edit_window=widgets.VBox([edit_lable,widgets.HBox([existing_points,import_point])])
    #edit_window.layout=widgets.Layout(top='15px',bottom='10px')

    content = widgets.VBox([coordinates_lables,inputs,widgets.HBox([notes,alignment]),apply_button,widgets.VBox(),out,edit_window])
    #content.layout=widgets.Layout(height='250px')
    children = [content,ww]
    tab = widgets.Tab()
    tab.children = children
    tab.set_title(0,'ADD NEW POINT')
    tab.set_title(1,'POINTS TABLE')

    return tab,points_objects,data



