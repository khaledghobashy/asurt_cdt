# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:41:13 2018

@author: khale
"""


import numpy as np
import scipy as sc

def jacobian_creator(topology):
    
    edgelist = topology.edges(data='obj')
    nodelist = topology.nodes
    
    n_nodes = len(nodelist)
    n_edges = len(edgelist)
            
    jacobian = np.zeros((n_edges+n_nodes,n_nodes),dtype=np.object)
    jacobian.fill(None)
    
    equations = np.zeros((n_edges+n_nodes,1),dtype=np.object)
    
    vel_rhs = np.zeros((n_edges+n_nodes,1),dtype=np.object)
    acc_rhs = np.zeros((n_edges+n_nodes,1),dtype=np.object)
    
    node_index = dict( (node,i) for i,node in enumerate(nodelist) )
        
    for ei,e in enumerate(edgelist):
        (u,v) = e[:2]
        eo    = e[2]
        
        ui = node_index[u]
        vi = node_index[v]
        
        if jacobian[ui+n_edges,ui]==None: jacobian[ui+n_edges,ui] = (u.jac,ui)
        if jacobian[vi+n_edges,vi]==None: jacobian[vi+n_edges,vi] = (v.jac,vi)
                
        jacobian[ei,ui] = (eo.jaci,ui)
        jacobian[ei,vi] = (eo.jacj,vi)
        
        equations[ei,0] = (eo.equations,ui,vi)
        
        vel_rhs[ei,0] = (eo.vel_rhs,ui,vi)
        
        acc_rhs[ei,0] = (eo.acc_rhs,ui,vi)
        
      
    def mapper(i,q):
        fun,ind = i
        return fun(q[ind])
    
    vectorized = np.vectorize(mapper,otypes=[np.object],excluded='q')
    
    return jacobian, equations, vel_rhs, acc_rhs, jacobian.nonzero(), vectorized


def jacobian_evaluator(jac_blocks,nzi,mapper,q):
    A = jac_blocks.copy()
    A[nzi]=mapper(A[nzi],q=q)
    return sc.sparse.bmat(A)
