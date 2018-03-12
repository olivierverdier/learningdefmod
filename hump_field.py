#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:41:13 2018

@author: barbara
"""



import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl
import odl
from odl.discr import Gradient

"""
Defines 1D field corresponding to humps
"""

def field_op(space, sigma):
    