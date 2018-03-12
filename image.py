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


""" 
Definition of geometrical descriptors that are images, defines the 
infinitesimal action

"""

def infinitesimal_action(v, I):
    grad_op = Gradient(I.space)
    grad = grad_op(I)
    return -sum(grad * grad)