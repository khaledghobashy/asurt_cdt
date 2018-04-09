# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:38:16 2018

@author: khaled.ghobashy
"""


import numpy as np
import pandas as pd

from base import ep2dcm


def body_dcm(dataframe,body):
    matrix_series = pd.DataFrame(np.zeros((len(dataframe),1)))
    print(matrix_series)
    for i in dataframe.T:
        print(i)
        p=dataframe.loc[i][body+'.e0':body+'.e3']
        dcm=ep2dcm(p)
        matrix_series.T[i]=dcm
    return matrix_series


def points_location_history(joints,position_df):
    joints_history_df = pd.DataFrame(columns=joints.index,index=position_df.index)
    for i in position_df.index:
        joints_history_df.loc[i] = [joints[k].joint_pos(position_df.loc[i]) for k in joints.index]
    
    return joints_history_df

