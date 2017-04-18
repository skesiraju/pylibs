#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 07 Dec 2015
# Last modified : 17 Feb 2015

"""
Classifiers
1. Gaussian Linear Classifier
"""

import sys
import numpy as np
import numexpr as ne
from .prep import standardize_data, standardize_test_data

# from memory_profiler import profile

def logsumexp(mat):
    """ log sum exp function, reasonably faster than scipy version """

    x_mat = np.copy(mat)
    xmax = np.max(x_mat, axis=0)
    ne.evaluate("exp(x_mat - xmax)", out=x_mat)
    x_mat = xmax + np.log(np.sum(x_mat, axis=0))
    inf_ix = ~np.isfinite(xmax)
    x_mat[inf_ix] = xmax[inf_ix]
    return x_mat


class GLC:
    """ Gaussian Linear Classifier """

    def __init__(self, est_prior=False, stdize=False):
        """
        Gaussian Linear classifier

        Parameters:
        -----------
        est_prior_type (bool): Estimate from training data or uniform\n
        stdize (bool): Standardize data, default = False\n
        """

        self.stdize = stdize
        self.est_prior = est_prior

        # label indices may not start from 0, hence this
        self.label_map = []

        self.w_k = None
        self.w_k0 = None
        self.log_post = None

        self.col_mean = None
        self.col_std = None

    def compute_cmus_scov(self, data, labels):
        """ Compute class means and shared covariance
        (weighted within-class covariance) """

        self.label_map, class_sizes = np.unique(labels, return_counts=True)
        dim = data.shape[1]

        cmus = np.zeros(shape=(len(class_sizes), dim))
        acc = np.zeros(shape=(dim, dim))
        scov = np.zeros_like(acc)

        gl_mu = np.mean(data, axis=0).reshape(-1, 1)
        gl_cov = np.cov(data.T, bias=True)

        for i, k in enumerate(self.label_map):
            data_k = data[np.where(labels == k)[0], :]
            mu_k = np.mean(data_k, axis=0).reshape(-1, 1)
            cmus[i, :] = mu_k[:, 0]
            acc += (gl_mu - mu_k).dot((gl_mu - mu_k).T) * data_k.shape[0]

        acc /= data.shape[0]
        scov = gl_cov - acc

        return cmus, scov, class_sizes

    def __check_test_data(self, test):
        """ Check the test data """

        if test.shape[1] != self.w_k.shape[1]:
            print("ERROR: Test data dimension is", test.shape[1],
                  "whereas train data dimension is", self.w_k.shape[1])
            print("Exiting..")
            sys.exit()

    def __compute_wk_and_wk0(self, class_mus, scov, priors):
        """ Return W_k which is a matrix, where every row corresponds to w_k
        for a particular class. Compute W_k0 which is a vector, where every
        element corresponds to w_k0 for a particular class """

        noc = class_mus.shape[0]
        self.w_k0 = np.zeros(shape=(noc), dtype=np.float64)

        s_inv = np.linalg.inv(scov)
        self.w_k = s_inv.dot(class_mus.T).T

        for k in range(noc):
            mu_k = class_mus[k].reshape(-1, 1)
            # self.w_k[k, :] = s_inv.dot(mu_k).reshape(dim)
            self.w_k0[k] = (-0.5 * (mu_k.T.dot(s_inv).dot(mu_k))) + priors[k]


    # @profile
    def train(self, data, labels):
        """ Train Gaussian linear classifier

        Parameters:
        -----------
        data (numpy.ndarray): row = data points, cols = dimension
        labels (numpy.ndarray): vector of labels

        Returns:
        --------
        Nothing
        """

        if self.stdize:
            data, self.col_mean, self.col_std = standardize_data(data)

        class_mus, scov, class_sizes = self.compute_cmus_scov(data, labels)
        noc = len(class_sizes)

        if self.est_prior:
            priors = class_sizes / noc
        else:
            priors = np.ones(shape=(noc, 1)) / noc

        # compute W_k (matrix), W_k0 (vector)
        self.__compute_wk_and_wk0(class_mus, scov, priors)

        # log posteriors
        self.log_post = self.w_k.dot(data.T).T
        np.add(self.w_k0, self.log_post, out=self.log_post)

    def predict(self, test, return_probs=False):
        """ Predict the class labels for the test data

        Parameters:
        -----------
        test (numpy.ndarray): rows = data points, cols = dimension

        Returns:
        --------
        prediction (numpy.ndarray): vector of predictions (labels)
        """

        self.__check_test_data(test)

        if self.stdize:
            test = standardize_test_data(test, self.col_mean, self.col_std)

        # For every test vector x, compute posterior of x belonging to class
        # C_k
        # P(C_k | x) = exp(a_k) / \sum_i{exp(a_i)}
        # where,
        #        a_k = W_k.T x + W_{0k}
        #        W_k = scov.inv() * mu_k
        #     W_{k0} = (-0.5)(mu.T)(scov.inv())(mu_k) + ln[P(C_k)]

        """
        y_p = np.zeros(shape=(test.shape[0], self.noc), dtype=float)
        for i in range(np.shape(test)[0]):

            x = test[i, :]
            x = x.reshape(self.dim, 1)

            a_k = np.zeros(shape=(self.noc), dtype=float)

            # compute likelihood of x, as belonging/generated by every class
            for k in range(self.noc):
                wk_T = self.w_k[k].reshape(1, self.dim)
                a_k[k] = wk_T.dot(x) + self.w_k0[k]  # posterior (not normalized)
                y_p[i, k] = a_k[k]
        """
        y_p = test.dot(self.w_k.T) + self.w_k0

        if return_probs:
            # y_p = y_p / np.sum(y_p, axis=1).reshape(-1, 1)
            # np.log(y_p, out=y_p)
            y_p = y_p.T
            y_p -= logsumexp(y_p)
            np.exp(y_p, out=y_p)
            y_p = y_p.T
        else:
            y_tmp = np.argmax(y_p, axis=1)
            # do the inverse label map
            y_p = [self.label_map[i] for i in y_tmp]

        return y_p

    def llh(self, test):
        """ Get the likelihoods instead of predictions

        Parameters:
        -----------
        test (numpy.ndarray): rows = data points, cols = dimension

        Returns:
        --------
        llh (numpy.ndarray): rows = data points, cols = llh for each class
        """

        pass

    def llh_ratios(self, test):
        """ Get the likelihood ratios instead of predictions

        Parameters:
        -----------
        test (numpy.ndarray): rows = data points, cols = dimension

        Returns:
        --------
        llh (numpy.ndarray): rows = data points, cols = llh ratios of the
        point belonging to the class w.r.t not belonging to the class
        """
        pass
