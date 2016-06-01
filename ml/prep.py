#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 07 Dec 2015
# Last modified : 17 Feb 2016

"""
Basic preprocessing machine learning utilities.
"""

import numpy as np


def standardize_data(data):
    """ Standardize the training data

    X = (X - mu) / (sigma)

    Parameters:
    -----------
    data: numpy ndarray, where rows are data points and cols are dimensions

    Returns:
    --------
    X_std (numpy array) \n
    col_mean (numpy array) \n
    col_std (numpy array)
    """
    col_mean = np.mean(data, axis=0)
    col_std = np.std(data, axis=0)
    sdata = np.subtract(data, col_mean) / col_std

    return sdata, col_mean, col_std


def standardize_test_data(X_test, col_mean, col_std):
    """ Standardize the test data using the stats from the training data """

    return np.subtract(X_test, col_mean) / col_std
