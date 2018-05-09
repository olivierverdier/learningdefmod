#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:40:38 2018

@author: barbara
"""



import numpy as np
from tensorflow.contrib import layers

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
        min_pt =[-10], max_pt=[10], shape=[32],
        dtype='float32', interp='linear')
points = np.transpose(space.points())
nb_points = points.shape[1]
m = nb_points

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
time_step = value = 1./N

#sigma for deformation
sigma_d = 1.
#deformation kernel in numpy
def kernel_d_np(x, y):
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

        right = np.argmax(GD_init)
        left = np.argmin(GD_init)

        for i in range(nb_int):
            new_position = np.zeros_like(GD_init)
            new_position[left] = GD_list[i][left] -  step * np.abs(speed)
            new_position[right] = GD_list[i][right] +  step * np.abs(speed)
            GD_list.append(new_position)

        return GD_list
    return integrate_trajectory
#



#%% example dimension 1

# GD = np.array([[0.], [5.]])

# vectors = np.ones([d, nb_points])
# structured0 = np.vstack([points, vectors])

integrate = integration_op(N, time_step)

def get_random_trajectory(speed):
    initial = np.zeros((2,1))
    initial = (10*(np.random.rand(2)) - 5.).reshape(2,1)
    trajectory = integrate(initial, speed)
    return trajectory


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

import tensorflow as tf

covariance = tf.constant(value=kernel_d_np(points.reshape(d, 1,-1), points.reshape(d, -1,1)))

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



# initialise GD
initial_descriptors = tf.placeholder(dtype=tf.float64, shape=[P, d], name="initial_descriptors")
data = tf.placeholder(dtype=tf.float64, shape=[N+1, P, d], name="data")


def field_network(z, reuse=True):
    """
    Network with input [None, P, d]
    Output of shape [None, k, m, d]
    """
    with tf.variable_scope('generator', reuse=reuse):
        z = tf.reshape(z, [1, -1]) # [-1, product of last dimensions of input]
        z = layers.fully_connected(z, num_outputs=16)
        z = layers.fully_connected(z, num_outputs=16)
        z = layers.fully_connected(z, num_outputs=16)
        z = layers.fully_connected(z, num_outputs=k*m*d, activation_fn=None)
        # # z = tf.reshape(z, [-1, 4, 4, 256])
        z = tf.reshape(z, [-1, m, k, d])

        # z = layers.conv2d_transpose(z, num_outputs=128, kernel_size=5, stride=2)
        # z = layers.conv2d_transpose(z, num_outputs=64, kernel_size=5, stride=2)
        # z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=2,
        #                             activation_fn=tf.nn.sigmoid)
        # return z[:, 2:-2, 2:-2, :]
        return z

field_network(initial_descriptors, reuse=False)

initial_cost = tf.constant(value=0., dtype=tf.float64)

velocity_cost_coefficient = tf.constant(value=1e-0, dtype=tf.float64)

step = tf.constant(value=time_step, dtype=tf.float64)

with tf.name_scope('cost'):
    descriptors = initial_descriptors
    cost = initial_cost
    for i in range(N):
        v_j = field_network(descriptors, reuse=True)
        cov_mat = make_covariance_matrix(points, descriptors)
        reshaped_vj = tf.reshape(v_j, (m, d)) # bad reshape!!
        velocity = tf.matmul(cov_mat, reshaped_vj)
        descriptors = descriptors + step * velocity
        current_cost = tf.norm(descriptors - data[i+1])
        cost = cost + current_cost
        transposed_reshaped_vj = tf.reshape(v_j, (d,m))
        velocity_cost = tf.squeeze(tf.matmul(transposed_reshaped_vj, tf.matmul(covariance, reshaped_vj)))
        cost = cost + velocity_cost_coefficient*velocity_cost

learning_rate = .1

with tf.name_scope('cost_summary'):
    tf.summary.scalar("cost", cost)


with tf.name_scope('adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

import datetime
import os

def get_writer():
    """
    Tensorboard writer based on run time
    """
    run_nb = '{:%H%M}'.format(datetime.datetime.now())
    print('run', run_nb)
    log_path = os.path.expanduser(os.path.join('~/tmp', 'defmod_logs',  'run_{}'.format(run_nb)))
    print(log_path)
    writer = tf.summary.FileWriter(log_path)
    return writer

