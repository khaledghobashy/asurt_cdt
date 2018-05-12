# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:53:57 2018

@author: khaled.ghobashy
"""

import os
os.chdir('..')

import networkx as nx
import pandas as pd
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt

from base import point, vector

class model(object):
    
    def __init__(self,name):
        
        self.data_graph = nx.DiGraph()
        self.topology   = nx.Graph()
        
    
    def add_point(self,name,coordinates,alignment):
        alignment_dict = {'S':'hps_','R':'hpr_','L':'hpl_'}
        if alignment =='S':
            p = point('hps_'+name,coordinates,alignment=alignment)
            self.data_graph.add_node(p.name,obj=p,typ='point')
            print('S')
        elif alignment in 'RL':
            name = alignment_dict[alignment]+name
            p1 = point(name,coordinates,alignment=alignment)
            p2 = p1.m_object
            self.data_graph.add_node(p1.name,obj=p1,typ='point')
            self.data_graph.add_node(p2.name,obj=p2,typ='point')
    
    def add_body(self,name):
        pass


m=model('m')
m.add_point('p',[1,-1,1],'L')
