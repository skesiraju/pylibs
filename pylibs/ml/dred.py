#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]om
# Date created : 07 Dec 2015
# Last modified : 17 Feb 2016

"""
Basic machine learning utilities for dimensionality reduction
1. Linear Discriminant Analysis
"""

import numpy as np
import sys
from prep import standardize_data, standardize_test_data
# from memory_profiler import profile


class LiDA:

    def __init__(self, data, labels, t_dim=-1):
        """
        Parameters:
        -----------
        t_dim (int): default is -1 which means same as the input dimension
        """

        self.X = data
        self.y = labels

        # no of data points, dim
        self.nop, self.dim = self.X.shape

        # no of classes
        self.labels = np.unique(self.y)
        self.noc = len(self.labels)

        if t_dim == -1:
            self.t_dim = self.dim
        else:
            if t_dim <= self.dim:
                self.t_dim = t_dim
            else:
                print('Error: Target dimension (%d) should be less than',
                      'input dimension (%d).' % t_dim, self.dim)
                print('Exiting..')
                sys.exit()

        self.W = None  # weights (eigen vectors or linear discriminant space)
        self.col_mean = None
        self.col_std = None

    def __compute_within_class_scatter(self):
        """ Compute with-in class scatter matrix """

        sel = [np.where(self.y == l) for l in self.labels]

        MU = np.zeros(shape=(self.noc, self.dim))  # mean for each class
        S_W = np.zeros(shape=(self.dim, self.dim))  # with-in class scatter

        for i in range(len(sel)):
            ci = sel[i]
            mu_i = np.mean(self.X[ci, :], axis=1)
            MU[i, :] = mu_i
            for j in ci[0]:
                x = self.X[j, :].reshape((self.dim, 1))
                m = mu_i.reshape((self.dim, 1))
                s_i = (x - m).dot((x - m).T)
                S_W += s_i

        return S_W

    def __compute_between_class_scatter(self):
        """ Compute between class scatter matrix """

        MU = np.mean(self.X, axis=0).reshape((self.dim, 1))
        sel = [np.where(self.y == l) for l in self.labels]

        S_B = np.zeros(shape=(self.dim, self.dim))  # between class scatter

        for i in range(len(sel)):
            cis = sel[i]  # class indices
            # mean of c_i
            mu_i = np.mean(self.X[cis, :], axis=1).reshape((self.dim, 1))
            s_b = len(cis[0]) * (mu_i - MU).dot((mu_i - MU).T)
            S_B += s_b

        return S_B

    def fit(self):
        """
        Given the data and labels, project the data on the linear discriminant
        space.
        """

        self.X, self.col_mean, self.col_std = standardize_data(self.X)
        S_W = self.__compute_within_class_scatter()
        S_B = self.__compute_between_class_scatter()

        if np.isfinite(np.linalg.cond(S_W)):

            eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

            eig_ixs = np.argsort(np.abs(eig_vals))
            top_eigs = eig_ixs[::-1][:self.t_dim]  # top 2 eig val indices

            self.W = np.empty(shape=(self.dim, len(top_eigs)))
            for i in range(len(top_eigs)):
                ix = top_eigs[i]
                self.W[:, i] = eig_vecs[:, ix].real
        else:
            # try SVD or LU decomposition
            print('Matrix is not invertible. This will be fixed later.')
            print('Exiting ..')
            sys.exit()

    def fit_transform(self):
        """
        Given the data and labels, project the data on the linear discriminant
        space and return the projected data. \n
        This is same as calling fit() followed by transform().

        Returns:
        --------
        numpy ndarray: projected data
        """

        self.fit()
        X_lda = self.X.dot(self.W)
        return X_lda

    def transform(self, X_test):
        """
        Project the data onto the linear discriminants that are obtained
        earlier (using fit method)

        Parameters:
        -----------
        X_test (numpy ndarray): rows are data points and cols are dim

        Returns:
        --------
        numpy ndarray: Projected data with same no. of points (rows) but same
        or less cols (dimensions)
        """

        X_test_std = standardize_test_data(X_test, self.col_mean,
                                           self.col_std)
        X_test_lda = X_test_std.dot(self.W)
        return X_test_lda
