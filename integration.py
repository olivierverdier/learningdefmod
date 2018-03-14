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
    """
    Operator integrating modular trajectories
    """
    
    def integrate_trajectory(GD_init, infinitesimal_action,
                             field_generator, control_trajectory, add_GD):
        """
        Returns the trajectory of GD as a list
        
        GD_init : initial geometrical descriptor
                    must have a copy function
        infinitesimal_action : function that takes in input a vector field and 
                               a geometrical descriptor GD and returns the 
                               application of the vector field to GD 
       field_generator : operators that takes in input a 
                               geometrical descriptor GD and a conrol and 
                               returns a vector field
       controlstrajectory : temporal list of nb_int controls
       add_GD : functions that takes ininput two GD and returns one GD
        
        """
        
        GD_list = []
        GD_list.append(GD_init.copy())
        
        for i in range(nb_int):
            field = field_generator(GD_list[i], step * control_trajectory[i])
            GD_list.append(add_GD(GD_list[i], infinitesimal_action(field, GD_list[i])))
            
        return GD_list
    return integrate_trajectory




def integration_op_listgen(nb_int, step):
    """
    Operator integrating modular trajectories from a list of generators
    corresponding to scalar controls
    
    """
    
    def integrate_trajectory(GD_init, infinitesimal_action,
                             field_generator_list, controls_trajectory,
                             mult_control_op, add_GD):
        """
        Returns the trajectory of GD as a list
        
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
       add_GD : functions that takes ininput two GD and returns one GD
        
        """
        
        GD_list = []
        GD_list.append(GD_init.copy())
        
        for i in range(nb_int):
            field_list = [field(GD_list[i]) for field in field_generator_list]
            field = mult_control_op(field_list,[step * cu for cu in controls_trajectory[i]])
            GD_list.append(add_GD(GD_list[i], infinitesimal_action(GD_list[i], field)))
            
        return GD_list
    return integrate_trajectory


