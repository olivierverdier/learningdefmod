#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:38:46 2018

@author: barbara
"""

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
Definition of geometrical descriptors that are lists of points, defines the 
infinitesimal action

"""

def infinitesimal_action(v, GD):
    GD = np.array(GD)
    speed = [im.interpolation(np.transpose(GD)) for im in v]
    return np.transpose(speed)