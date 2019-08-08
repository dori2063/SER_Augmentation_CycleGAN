#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:17:39 2019

@author: Youngdo Ahn
"""

import numpy as np

n_img_rows = 64#40#113
n_img_cols = 32#40#14


def real_feat_extra(x):
    window=np.zeros(n_img_rows*n_img_cols)
    window[233:1582+233] = 1
    x = x*window
    return x

def real_feat_shape(input_shape):
    return input_shape