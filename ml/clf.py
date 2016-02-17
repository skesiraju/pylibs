#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 07 Dec 2015
# Last modified : 17 Feb 2015

"""
Classifiers
1. Linear Gaussian Classifier
"""

from __future__ import print_function
import numpy as np
import sys
from prep import standardize_data, standardize_test_data
# from memory_profiler import profile


class GC:

    def __init__(self, est_prior=True, stdize=False, check=False):
        """
        Linear Gaussian classifier

        Parameters:
        -----------
        est_prior_type (bool): Estimate from training data or uniform
        stdize (bool): Standardize data, default = False
        check (bool): Cross check ACC, WCC computations
        """

        self.NOC = None
        self.DIM = None
        self.stdize = stdize
        self.check = check
        self.est_prior = est_prior

        # label indices may not start from 0, hence this
        self.label_map = []

        self.W_k = None
        self.W_k0 = None

        self.col_mean = None
        self.col_std = None

    def __compute_WCC(self, feat_mat):
        """ Compute with-in class covariance """

        mu = np.mean(feat_mat, axis=0, dtype=np.float64)  # mean for all feats
        mu = np.reshape(mu, (len(mu), 1))
        r, c = np.shape(feat_mat)
        cov = np.zeros(shape=(self.DIM, self.DIM), dtype=np.float64)
        for i in xrange(r):
            vec = np.reshape(feat_mat[i, :], (len(feat_mat[i, :]), 1))
            cov += np.dot((vec - mu), (vec - mu).T)

        cov = np.divide(cov, np.float64(r))

        return mu, cov

    def __compute_ACC(self, MU, class_mu, class_size, N):
        """ Compute across class covariance """

        cov_ac = np.zeros(shape=(self.DIM, self.DIM), dtype=np.float64)
        for i in xrange(len(class_mu)):
            mu_c = np.reshape(class_mu[i, :], (len(class_mu[i, :]), 1))
            cov_ac += np.multiply(np.dot((mu_c - MU), (mu_c - MU).T),
                                  class_size[i])

        cov_ac = np.divide(cov_ac, np.float64(N - 1))

        return cov_ac

    def __check(self, class_cov, cov_ac, gl_cov, class_sizes, gl_mu,
                class_mus, N):
        """ Cross check the cov computation """

        flag = False

        ccov = np.zeros(shape=(self.DIM, self.DIM), dtype=np.float64)
        for i in xrange(len(class_cov)):
            ccov += np.multiply(class_cov[i], class_sizes[i])
            ccov = np.divide(ccov, np.float64(N - 1))
        ccov += cov_ac

        weights = np.divide(class_sizes, N)
        wt_avg_cmu = np.average(class_mus, axis=0, weights=weights)
        wt_avg_cmu = np.reshape(wt_avg_cmu, (self.DIM, 1))

        if np.allclose(gl_mu, wt_avg_cmu):
            print('Cross checked the mean computation. Good to go.')
        else:
            print('Something is wrong in mean computation.')
            flag = True

        if np.allclose(gl_cov, ccov):
            print('Cross checked the covariance computation. Good to go.')
        else:
            print('Something is wrong in WCC, ACC computations.')
            flag = True

        if flag:
            print('Exiting..')
            sys.exit()

    def __check_test_data(self, test):
        """ Check the test data """

        nop, dim = test.shape
        if dim != self.DIM:
            print("ERROR: Test data dimension is", dim,
                  "whereas train data dimension is", self.DIM)
            print("Exiting..")
            sys.exit()

    def __compute_W_k_and_W_k0(self, class_mus, scov, priors):
        """ Return W_k which is a matrix, where every row corresponds to w_k
        for a particular class. Compute W_k0 which is a vector, where every
        element corresponds to w_k0 for a particular class """

        self.W_k = np.zeros(shape=(self.NOC, self.DIM), dtype=np.float64)
        self.W_k0 = np.zeros(shape=(self.NOC), dtype=np.float64)

        s_inv = np.linalg.inv(scov)

        for i in xrange(self.NOC):
            mu = class_mus[i].reshape(self.DIM, 1)
            self.W_k[i, :] = s_inv.dot(mu).reshape(self.DIM)

            self.W_k0[i] = (-0.5 * (mu.T.dot(s_inv).dot(mu))) + priors[i]

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

        self.label_map = sorted(np.unique(labels))

        self.NOC = len(self.label_map)
        N, self.DIM = data.shape

        class_sizes = np.zeros(shape=(self.NOC), dtype=np.float64)
        weights = np.zeros(shape=(self.NOC), dtype=np.float64)

        class_mus = np.zeros(shape=(self.NOC, self.DIM), dtype=np.float64)
        class_covs = np.zeros(shape=(self.NOC, self.DIM, self.DIM),
                              dtype=np.float64)

        # k is index for class labels, because the labels may not
        # start from 0
        k = 0
        for i in self.label_map:
            ixs = np.where(labels == i)[0]
            data_ci = data[ixs]

            class_sizes[k] = len(ixs)
            mu_k, cov_k = self.__compute_WCC(data_ci)

            class_mus[k, :] = mu_k.T
            class_covs[k, :, :] = cov_k
            k += 1

        # global mean
        gl_mu = np.reshape(np.mean(data, axis=0, dtype=np.float64),
                           (self.DIM, 1))

        cov_ac = self.__compute_ACC(gl_mu, class_mus, class_sizes, N)

        if self.check:
            # global cov
            gl_cov = np.cov(data.T)
            self.__check(class_covs, cov_ac, gl_cov, class_sizes, gl_mu,
                         class_mus, N)

        if self.est_prior:
            priors = weights
        else:
            priors = np.asarray([1.0 / self.NOC] * self.NOC)

        # shared COV
        scov = np.zeros(shape=(self.DIM, self.DIM), dtype=np.float64)
        for i in xrange(self.NOC):
            scov += np.multiply(class_covs[i], class_sizes[i])
        scov = np.divide(scov, np.float64(N))

        # compute W_k (matrix), W_k0 (vector)
        self.__compute_W_k_and_W_k0(class_mus, scov, priors)

    def predict(self, test):
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

        y_p = np.zeros(shape=(test.shape[0]))
        for i in xrange(np.shape(test)[0]):

            x = test[i, :]
            x = x.reshape(self.DIM, 1)

            a_k = np.zeros(shape=(self.NOC), dtype=np.float64)

            # compute likelihood of x, as belonging/generated by every class
            for k in xrange(self.NOC):
                wk_T = self.W_k[k].reshape(1, self.DIM)
                lh = wk_T.dot(x) + self.W_k0[k]  # posterior (not normalized)
                a_k[k] = lh

            y_p[i] = self.label_map[np.argmax(a_k)]  # test predicted

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
