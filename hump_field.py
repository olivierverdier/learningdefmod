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
Defines field and images corresponding to humps
"""

__all__ = ('field_op', 'image_op')

def field_op(space, sigma):
    def kernel(x, y):
        return np.exp(-sum([(xi - yi)**2 for xi, yi in zip(x, y)]) / (sigma ** 2))
    
    mg = space.meshgrid
    def generate_hump_field(GD, Cont):
        """
        Returns vector field obtained by summing local translations centred at 
        GDs, with vectors in Cont and localized by kernel.
        
        GD : array, size : nb_points x dim
        Cont : array, size : nb_points x dim
        """
        
        nb_points, dim = GD.shape
        field = space.tangent_bundle.zero()

        for k in range(nb_points):
            def kern_app_point(x):
                return kernel(x, GD[k])

            kern_discr = kern_app_point(mg)

            field += space.tangent_bundle.element([kern_discr * vect for vect in Cont[k]]).copy()

        return field

    return generate_hump_field




def image_op(space, sigma):
    def kernel(x, y):
        return np.exp(-sum([(xi - yi)**2 for xi, yi in zip(x, y)]) / (sigma ** 2))
    
    mg = space.meshgrid
    def generate_hump_image(GD, Coeff):
        """
        
        Returns image obtained by summing gaussians centred at GDs and 
        with coefficients in Coeff.
        
        GD : array, size : nb_points x dim
        Coeff : array, size : nb_points x 1
        """
        
        nb_points, dim = GD.shape
        im = space.zero()

        for k in range(nb_points):
            def kern_app_point(x):
                return kernel(x, GD[k])

            kern_discr = kern_app_point(mg)

            im += space.element(kern_discr * Coeff[k]).copy()

        return im

    return generate_hump_image
#