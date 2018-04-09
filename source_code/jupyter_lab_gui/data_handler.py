# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:20:41 2018

@author: khale
"""

import pandas as pd
from base import point


def points_creator(points_csv_file):
    
    # converting csv file into dataframe
    data_frame = pd.read_csv(points_csv_file)
    data_frame = (data_frame.T['Name':]).T
    
    points_series = pd.Series()
    
    for i in data_frame.index:
        name  = 'hpr_'+data_frame.loc[i]['Name']
        x,y,z = data_frame.loc[i]['X':'Z']
        points_series[name]=point(name,[x,y,z])
        
        if data_frame.loc[i]['Mirrored']==True:
            name  = 'hpl_'+data_frame.loc[i]['Name']
            x,y,z = data_frame.loc[i]['X':'Z']
            points_series[name]=point(name,[x,-y,z])
    
    return points_series

