#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:28:12 2018

@author: barbara
"""



import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl
import odl
from odl.discr import Gradient
from odl.deform.linearized import _linear_deform


""" 
Definition of geometrical descriptors that are images, defines the 
infinitesimal action

"""

def infinitesimal_action_withgrad(v, I):
    grad_op = Gradient(I.space)
    grad = grad_op(I)
    return -sum(v * grad)


def infinitesimal_action(v, I):
    step = 0.1
    I_tmp=I.space.element(_linear_deform(I, -step* v).copy())
    return (1/step) * (I_tmp - I)



        


