#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:40:38 2018

@author: barbara
"""



import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl

import numpy as np
import odl 

import sys
sys.path.insert(0, '/home/barbara/learningdefmod')
import fun_generate_data as fun_gen

#%% Parameters : 
#dimension
d = 1

#grid points
space = odl.uniform_discr(
        min_pt =[-10], max_pt=[10], shape=[56],
        dtype='float32', interp='linear')
points = np.transpose(space.points())
nb_points = points.shape[1]

##dimension
#d = 2
#
##grid points
#space = odl.uniform_discr(
#        min_pt =[-10., -10.], max_pt=[10, 10], shape=[56],
#        dtype='float32', interp='linear')
#points = np.transpose(space.points())
#nb_points = points.shape[1]



#number of vector fields
k = 1

#number of image points
P = 2

#number of time points
N = 10
step = 1. / N

#sigma for deformation
sigma_d = 1.
#deformation kernel in numpy
def kernel_d_np(x, y):
    #si = tf.shape(x)[0]
    return np.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(d)]) / (sigma_d ** 2))



#%%  Generate data from GD

##dimension 1
#def infinitesimal_action(structured_field, GD):
#    """
#    returns the application of the field to points of GD
#    
#    GD : geometrical descriptor (made of points)
#        array of size P x d
#    structured fields : field (discretized on grid)
#         array of size 2*d x number of grid points
#         
#    """
#
#    points = structured_field[0:d]
#    vectors = structured_field[d::]
#
#    GD_reshaped = np.reshape(GD, [1, 1 , P])
#    
#    points_reshaped = np.reshape(points, [1, nb_points, 1])
#    
#    kern_discr = kernel_d_np(GD_reshaped, points_reshaped)
#    kern_discr.shape
#    
#    speed = (np.transpose(vectors) * kern_discr).sum(1)
#    
#    return np.reshape(speed, [nb_points, d])
##

#integration operator for the example where GD is made of points which all 
#have the same speed, and that it can be modelled by a constant control
#equal to one : the speed is constant over time and positive for all data
def integration_op(nb_int, step):
    """
    Operator integrating modular trajectories
    """
    
    def integrate_trajectory(GD_init, speed):
        """
        Returns the trajectory of GD (which are points) as a list
        
        GD_init : initial geometrical descriptor
                array of size P x d
       speed_trajectory : one vector (in dimension 1)
       
        
        """
        
        GD_list = []
        GD_list.append(GD_init.copy())
        
        for i in range(nb_int):
            GD_list.append(GD_list[i] +  step * np.abs(speed))
            
        return GD_list
    return integrate_trajectory
#



#%% example dimension 1

GD = np.array([[0.], [5.]])

vectors = np.ones([d, nb_points])
structured0 = np.vstack([points, vectors])

integrate = integration_op(N, step)

speed = np.random.rand()
if (speed > min(10 - max(GD)[0], min(GD)[0] + 10)):
    print('speed to big : gets out of space')
GD_list = integrate(GD, speed)


# remark : attention, to generate data speed must be constant for all the 
# data set, in order to make sure it is not to big, constrain GD to be 
# close enough to 0


#
##%% example dimension 2
#
#GD = np.array([[0., -10.], [10., 1.]])
#
#vectors = np.ones([d, nb_points])
#structured0 = np.vstack([points, vectors])
#


#%% Cost function

#tf.contrib.image.transform
def kernel(x, y):
    #si = tf.shape(x)[0]
    return tf.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(d)]) / (sigma_d ** 2))

def make_covariance_matrix(points1, points2):
    """ creates the covariance matrix of the kernel for the given points"""

    #dim = tf.shape(points)[0]
    p1 = tf.reshape(points1, (d, 1, -1))
    p2 = tf.reshape(points2, (d, -1, 1))

    return kernel(p1, p2)

def create_structured(points, vectors):
    return tf.concat([points, vectors], 0)

v_j = tf.Variable(np.ones([1, nb_points]), name="v_j")

structured_list_computed = create_structured(points, v_j)

cov_mat = make_covariance_matrix(points, GD)
speed = tf.matmul(cov_mat, tf.transpose(v_j))

# integration
GD_init = tf.Variable(np.ones([P, d]), name="GD_init")
shape = tf.constant([N,P])
GD_list = tf.zeros(shape, dtype='float64')
indices = tf.constant([[0]])
GD_list += tf.scatter_nd(indices, tf.transpose(GD_init), shape)

tf.while_loop(i<N, i+=1)


#%% launch

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

alpha_init = np.ones([1, nb_points])
alpha = alpha_init.copy()

structured_list_computed.eval(feed_dict={v_j: alpha_init})


GD_list.eval(feed_dict={GD_init: GD})

