# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:54:24 2018

@author: khale
"""


import pandas as pd
import numpy as np
import qgrid
import sys

points_df = pd.DataFrame({
    'Name' : 'Point',
    'X'    : np.array([0]*1),
    'Y'    : np.array([0]*1),
    'Z'    : np.array([0]*1)})

points_df['Mirrored']= True
points_df['Notes']= '.......'

#points_df = (points_df.T['Name':]).T

grid_options = {
    'fullWidthRows': False,
    'syncColumnCellResize': True,
    'forceFitColumns': False,
    'defaultColumnWidth': 100,
    'rowHeight': 28,
    'enableColumnReorder': False,
    'enableTextSelectionOnCells': True,
    'editable': True,
    'autoEdit': False,
    'explicitInitialization': True,
    'maxVisibleRows': 10,
    'minVisibleRows': 8,
    'sortable': False,
    'filterable': True,
    'highlightSelectedCell': True,
    'highlightSelectedRow': True
}



points_grid = qgrid.QgridWidget(df=points_df, grid_options=grid_options,show_toolbar=True)
points_grid
