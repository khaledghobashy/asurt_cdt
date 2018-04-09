
from __future__ import print_function
import ipywidgets as widgets
import pandas as pd
import sys

point_name = widgets.Label(value=' Name')
x = widgets.Label(value='$X$')
y = widgets.Label(value='$Y$')
z = widgets.Label(value='$Z$')
title = widgets.HBox([point_name,x,y,z])

name = widgets.Text(placeholder='Point Name')
x = widgets.FloatText()
y = widgets.FloatText()
z = widgets.FloatText()
d = widgets.Textarea(placeholder='Optional brief describtion.')
x.layout=y.layout=z.layout=widgets.Layout(width='100px')
name.layout=widgets.Layout(width='120px')


inputs = widgets.HBox([name,x,y,z])

render = widgets.VBox([title,inputs,inputs])
render.children=render.children+(inputs,)

add = widgets.Button(description='Apply')

out = widgets.Output()


points=pd.Series()
def added(b):
    out.clear_output()
    with out:
        p=point(name.value,[x.value,y.value,z.value])
        points[name.value]=p
        name.value=''
        print('Done')
        

    
add.on_click(added)


tab_contents = ['P0', 'P1', 'P2', 'P3', 'P4']
children = [widgets.VBox([inputs,d,add,out]),inputs]
tab = widgets.Tab()
tab.children = children
tab.set_title(0,'ADD NEW POINT')
tab.set_title(1,'EDIT EXISTING POINTS')
tab.selected_index=0
sys.stdout(tab)