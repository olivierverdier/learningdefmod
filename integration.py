#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:29:07 2018

@author: barbara
"""



import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl


def integration_op(nb_int, step):
    def integrate_trajectory(GD_init, infinitesimal_action,
                             field_generator_list, controls_trajectory,
                             mult_control_op):
        """
        GD_init : initial geometrical descriptor
                    must have a copy function
        infinitesimal_action : function that takes in input a vector field and 
                               a geometrical descriptor GD and returns the 
                               application of the vector field to GD 
       field_generator_list : list of operators that take in input a 
                               geometrical descriptor GD and returns a 
                               vector field
       controls_trajectory : temporal list of nb_int controls, each one being
                             a list of scalar with same size as 
                             field_generator_list
       mult_control_op : operator that takes in input a list of vector field 
                            and a list of scalar controls (same size of list)
                            and returns the linear combination of the vector
                            fields with the scalar controls as coefficients
        
        """
        
        GD_list = []
        GD_list.append(GD_init.copy())
        
        for i in range(nb_int):
            field_list = [field(GD_list[i]) for field in field_generator_list]
            field = mult_control_op(field_list, controls_trajectory[i])
            GD_list.append(infinitesimal_action(GD_list[i], field))
            
        return GD_list
    return integrate_trajectory


