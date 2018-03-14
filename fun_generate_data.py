#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:23:31 2018

@author: barbara
"""


def generate_data_fromgd(GD_init, infinitesimal_action,
                             field_op, control_trajectory, data_op,
                             integration_op, add_GD):
    """
    Integrates values of GD and then generate the list of data from them.
    We suppose here that data (ex image) can be defined from  GD
    
    GD_init : initial geometrical descriptor
    infinitesimal_action : function that takes in input a vector field and 
                       a geometrical descriptor GD and returns the 
                       application of the vector field to GD 
     field_op : operators that takes in input a 
                       geometrical descriptor GD and a conrol and 
                       returns a vector field
     controls_trajectory : temporal list of nb_int controls, each one being
                       list of scalar with same size as 
                       field_generator_list
    data_op : function that takes in input a GD and returns a data (ex : image)
    integration_op : takes in input (GD_init, infinitesimal_action,
                             field_op, control_trajectory, add_GD)
                    and returns the trajectory of GD as a list
       add_GD : functions that takes ininput two GD and returns one GD
    
    """
    
    GD_list = integration_op(GD_init, infinitesimal_action,
                             field_op, control_trajectory, add_GD)
    image_list = [data_op(GDi).copy() for GDi in GD_list]
    
    return [GD_list.copy(), image_list.copy()]


def generate_data_fromaction(GD_init, infinitesimal_action,
                             field_op, control_trajectory, add_GD, data_init, 
                             infinitesimal_action_data, add_data,
                             integration_op):
    """
    Integrates values of GD and generate the list of data from 
    an initial one by integrating
    
    GD_init : initial geometrical descriptor
    infinitesimal_action : function that takes in input a vector field and 
                       a geometrical descriptor GD and returns the 
                       application of the vector field to GD 
     field_op : operators that takes in input a 
                       geometrical descriptor GD and a conrol and 
                       returns a vector field
     controls_trajectory : temporal list of nb_int controls, each one being
                       list of scalar with same size as 
                       field_generator_list
    data_init : initial data
    infinitesimal_action_data : function that takes in input a vector field and 
                       a data and returns the 
                       application of the vector field to the data  
    integration_op : takes in input (GD_init, infinitesimal_action,
                             field_op, control_trajectory)
                    and returns the trajectory of GD as a list
       add_GD : functions that takes in input two GD and returns one GD
       add_data : functions that takes in input two data and returns one data
    
    """
    
    GD_compound_init = [GD_init, data_init]
    def infinitesimal_action_compound(field, GD_compound):
        return [infinitesimal_action(field, GD_compound[0]),
                infinitesimal_action_data(field, GD_compound[1])]
    def field_op_compound(GD_compound, Cont):
        return field_op(GD_compound[0], Cont)
    def add_compound(GD_compound0, GD_compound1):
        return[add_GD(GD_compound0[0], GD_compound1[0]), 
               add_data(GD_compound0[1], GD_compound1[1])]
    
    GD_compund_list = integration_op(GD_compound_init, infinitesimal_action_compound,
                             field_op_compound, control_trajectory, add_compound)
    
    return [[GD[0] for GD in GD_compund_list], [GD[1] for GD in GD_compund_list]]
