#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:21:15 2018

@author: barbara
"""

"""
Created on Fri Mar  9 16:41:13 2018

@author: barbara
"""



import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl
import odl
from odl.discr import Gradient
import hump_field
import image
import integration
import points
import fun_generate_data
#%%
space = odl.uniform_discr(
        min_pt =[-10], max_pt=[10], shape=[512],
        dtype='float32', interp='linear')

sigma0 = 1
sigma1 = 1
nb_int = 10
step = 1/nb_int
field_op = hump_field.field_op(space, sigma0)
image_op = hump_field.image_op(space, sigma1)
integration_op = integration.integration_op(nb_int, step)
#%%
#
GD_init = np.array([[-2], [3]])
Coeff = np.array([[2], [3]])

def data_op(GD):
    return image_op(GD, Coeff)
#GD_init = np.array([ [-3]])
#Coeff = np.array([[1]])

template = data_op(GD_init)
template.show()

#%%
infinitesimal_action = points.infinitesimal_action

#%%

control_trajectory = [np.array([[0], [5]]) for i in range(nb_int)]

def add_GD(GD0, GD1):
    return GD0 + GD1

GD_list = integration_op(GD_init, infinitesimal_action,
                             field_op, control_trajectory, add_GD)

#%%
GD_list, image_list = fun_generate_data.generate_data_fromgd(GD_init, infinitesimal_action,
                             field_op, control_trajectory, data_op,
                             integration_op, add_GD)
 
#GD_list = fun_generate_data.generate_data_fromaction(GD_init, infinitesimal_action,
#                             field_op, control_trajectory, add_GD, GD_init,
#                             infinitesimal_action, add_GD,
#                             integration_op)
  
#%%
if False:
    for i in range(nb_int + 1):
        image_list[i].show(str(i))


#%% Problem with infinitesimal action by gradient
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

image_list = []
image_list.append(template)
field_list = []
for i in range(nb_int):
    field = field_op(GD_list[i], control_trajectory[i])
    field_list.append(field.copy())
    #field.show(str(i))
    image_list.append(space.element(image_list[i] + step  * image.infinitesimal_action(field, image_list[i])))
#%%

for i in range(nb_int+1):
    #field_list[i].show('field' + str(i))
    image_list[i].show(str(i))
