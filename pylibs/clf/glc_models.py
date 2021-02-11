#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cSpell: disable

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 15 Feb 2018
# Last modified : 15 Feb 2018

"""
Gaussian Linear Classifier with Uncertainty (GLCU)
"""

import os
import sys
import shutil
import argparse
import tempfile
import numpy as np
import scipy.sparse as sparse
from scipy.special import logsumexp as lse
import numexpr as ne


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

    def __init__(self, est_prior=False):
        """
        Gaussian Linear classifier

        Parameters:
        -----------
            est_prior_type (bool): Estimate from training data or uniform
        """

        self.est_prior = est_prior

        # label indices may not start from 0, hence this
        self.label_map = []

        self.w_k = None
        self.w_k0 = None
        self.log_post = None

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

    def __compute_wk_and_wk0(self, class_mus, scov, log_priors):
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
            self.w_k0[k] = (-0.5 * (mu_k.T.dot(s_inv).dot(mu_k))) + log_priors[k]

    def train(self, data, labels):
        """ Train Gaussian linear classifier

        Parameters:
        -----------
            data (numpy.ndarray): row = data points, cols = dimension
            labels (numpy.ndarray): vector of labels
        """

        class_mus, scov, class_sizes = self.compute_cmus_scov(data, labels)
        noc = len(class_sizes)

        if self.est_prior:
            # print('Using class priors.')
            log_priors = np.log(class_sizes / class_sizes.sum())
        else:
            # print('Uniform class priors.')
            log_priors = np.log(np.ones(shape=(noc, 1)) / noc)

        # compute W_k (matrix), W_k0 (vector)
        self.__compute_wk_and_wk0(class_mus, scov, log_priors)

        # log posteriors
        self.log_post = self.w_k.dot(data.T).T
        np.add(self.w_k0, self.log_post, out=self.log_post)

    def predict_train(self):
        """ Predict the training set and return the labels """

        y_tmp = np.argmax(self.log_post, axis=1)
        # do the inverse label map
        return [self.label_map[i] for i in y_tmp]

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

        # For every test vector x, compute posterior of x belonging to class
        # C_k
        # P(C_k | x) = exp(a_k) / \sum_i{exp(a_i)}
        # where,
        #        a_k = W_k.T x + W_{0k}
        #        W_k = scov.inv() * mu_k
        #     W_{k0} = (-0.5)(mu.T)(scov.inv())(mu_k) + ln[P(C_k)]


        # y_p = np.zeros(shape=(test.shape[0], self.noc), dtype=float)
        # for i in range(np.shape(test)[0]):

        #    x = test[i, :]
        #    x = x.reshape(self.dim, 1)

        #    a_k = np.zeros(shape=(self.noc), dtype=float)

        #    # compute likelihood of x, as belonging/generated by every class
        #    for k in range(self.noc):
        #        wk_T = self.w_k[k].reshape(1, self.dim)
        #        a_k[k] = wk_T.dot(x) + self.w_k0[k]  # posterior (not normalized)
        #        y_p[i, k] = a_k[k]

        y_p = test.dot(self.w_k.T) + self.w_k0

        y_tmp = np.argmax(y_p, axis=1)
        # do the inverse label map
        y_p = [self.label_map[i] for i in y_tmp]

        return np.asarray(y_p, dtype=int)

    def predict_proba(self, test):
        """ Predict the posterior probabilities of class labels for the test data

        Parameters:
        -----------
            test (numpy.ndarray): rows = data points, cols = dimension

        Returns:
        --------
            prediction (numpy.ndarray): Posterior probabilites of class
        """

        self.__check_test_data(test)

        y_p = test.dot(self.w_k.T) + self.w_k0

        y_p = y_p.T
        y_p -= logsumexp(y_p)
        np.exp(y_p, out=y_p)
        y_p = y_p.T

        return y_p


class GLCU:
    """ Gaussian Linear Classifier with uncertainty """

    def __init__(self, trn_iters=5, cov_type='diag', est_prior=False):
        """ Initialize model configuration """

        if cov_type != "diag":
            print("cov_type should be diag. Others not implemented.")
            sys.exit()

        self.cov_type = cov_type
        self.trn_iters = trn_iters
        self.est_prior = est_prior

        self.cmus = None
        self.scov = None
        self.m2l = None
        self.c_sizes = None
        self.dim = None
        self.priors = None

        self.tmp_dir = tempfile.mkdtemp() + "/"

    def compute_cmus_shared_cov(self, means, labels):
        """ Compute class means and shared covariance matrix """

        _, self.c_sizes = np.unique(labels, return_counts=True)
        self.c_sizes = self.c_sizes.astype(np.float32)
        noc = len(self.c_sizes)

        if self.est_prior:
            self.priors = (self.c_sizes / self.c_sizes.sum()).astype(np.float32)
        else:
            self.priors = np.ones(shape=(noc, 1), dtype=np.float32) / noc

        N = means.shape[0]  # number of samples
        # means 2 label mapping
        self.m2l = sparse.csc_matrix((np.ones(N), (range(N), labels)),
                                     dtype=np.float32)
        self.cmus = self.m2l.T.dot(means) / self.c_sizes.reshape(-1, 1)

        # -- sanity check --
        # global_mu = np.mean(ivecs, axis=0).reshape(1, -1)
        # global_cov = np.cov(ivecs.T, bias=True)
        # tmp = (class_mus - global_mu) * np.sqrt(class_sizes.reshape(-1, 1))
        # acc = tmp.T.dot(tmp) / N
        # shared_cov = global_cov - acc  # OR weighted with-in class cov

        tmp = means - self.m2l.dot(self.cmus)
        self.scov = (tmp.T.dot(tmp) / N).astype(np.float32)

    def train_sb(self, data, labels, covs=None):
        """ Train using EM

        Args:
            data (np.ndarray): If cov_type is `diag` then, shape is n_docs x [twice dim]
            (every row is twice the dimension of latent variable, where the
             first half is mean, and second half is log std.dev). Otherwise data
             represents only mus of shape n_docs x dim.
            labels (np.ndarray): Labels for n_docs. Shape is n_docs
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
            then data represents only mus and covs are passed as parameters
        """

        if min(labels) == 1:
            labels -= 1

        if self.cov_type == 'diag':
            self.dim = int(data.shape[1] / 2)
            mus = data[:, :self.dim]
            covs = np.exp(2 * data[:, self.dim:])

        else:
            self.dim = data.shape[1]
            mus = data

        self.compute_cmus_shared_cov(mus, labels)

        gammas = 1. / covs

        eye = np.eye(mus.shape[1], dtype=np.float32)  # Identity matrix

        spre = np.linalg.inv(self.scov)  # shared precision

        # Parameters of the posterior distribution of latent variables
        #  p(Z|W,Mus,S)
        alphas = np.zeros_like(mus, dtype=np.float32)
        lambdas = np.zeros(shape=(mus.shape[0], self.dim, self.dim),
                           dtype=np.float32)

        for _ in range(self.trn_iters):

            # E - step: get the posterior of latent variables Z
            # i.e, q(z_n) = N(z_n | alpha_n, Lambda_n.inv())
            # alpha_n = [I + D.inv().dot(Gamma_n)].inv().dot(w_n - mu_n)
            # Lambda_n = D + Gamma_n

            tmp = mus - self.m2l.dot(self.cmus)
            if self.cov_type == 'diag':
                for n in range(mus.shape[0]):
                    # alphas[n, :] = np.linalg.inv((self.scov * gammas[n, :])
                    #                             + eye).dot(tmp[n, :])
                    alphas[n, :] = np.linalg.solve((self.scov * gammas[n, :])
                                                   + eye, tmp[n, :])
                    lambdas[n, :, :] = spre + np.diag(gammas[n, :])
            else:
                for n in range(mus.shape[0]):
                    # alphas[n, :] = np.linalg.inv(self.scov.dot(gammas[n, :, :])
                    #                             + eye).dot(tmp[n, :])
                    alphas[n, :] = np.linalg.solve(self.scov.dot(gammas[n, :, :])
                                                   + eye, tmp[n, :])

                lambdas = gammas + spre

            # M - step: maximize w.r.t params (class mus, shared precision matrix)
            # M.a maximize w.r.t. class mus
            # mu_l = (1/N_l) \sum_{n in l} (w_n - alpha_n)

            self.cmus = self.m2l.T.dot(mus - alphas) / self.c_sizes.reshape(-1, 1)

            # M.b maximize w.r.t. shared cov (precision) matrix
            # S = 1/N [ (\sum_n \Lambda_n.inv()) +
            #           (alpha_n - (w_n - mu_l)) (alpha_n - (w_n - mu_l)).T ]

            tmp = alphas - (mus - self.m2l.dot(self.cmus))
            self.scov = (np.linalg.inv(lambdas).sum(axis=0) +
                         tmp.T.dot(tmp)) / mus.shape[0]
            spre = np.linalg.inv(self.scov)



    def e_step(self, mus, covs, i, bsize):
        """ E step - get the posterior distribution of latent variables.

        Args:
        -----
            mus (np.ndarray): means of data points or embeddings
            covs (np.ndarray): covs of data points or embeddings
            i (int): training iteration number
            bsize (int): batch size

        Returns:
        --------
            np.ndarray (alphas): Means of posterior dist. of latent vars.
        """

        eye = np.eye(mus.shape[1], dtype=np.float32)  # Identity matrix

        gammas = 1. / covs

        spre = np.linalg.inv(self.scov).astype(np.float32)  # shared precision

        # Parameters of the posterior distribution of latent variables
        # p(Z|W,Mus,S) = N(Z | alphas, Lambdas)

        alphas = np.zeros_like(mus, dtype=np.float32)

        # E - step: get the posterior of latent variables q(z_n) ~ N(alpha_n, Lambda_n.inv())
        # alpha_n = [I + D.inv().dot(Gamma_n)].inv().dot(w_n - mu_n)
        # Lambda_n = D + Gamma_n

        tmp = mus - self.m2l.dot(self.cmus)

        sdoc = 0
        edoc = bsize

        bno = 0  # batch number
        while sdoc < edoc:

            lambdas_batch = np.zeros(shape=(edoc-sdoc, self.dim, self.dim),
                                     dtype=np.float32)
            # print('\r    e-step batch num {:2d}/{:2d}'.format(bno, mus.shape[0]//bsize),
            #      end=" ")

            if self.cov_type == 'diag':

                for n in range(sdoc, edoc, 1):
                    alphas[n, :] = np.linalg.solve((self.scov * gammas[n, :])
                                                   + eye, tmp[n, :])
                    lambdas_batch[n-sdoc, :, :] = spre + np.diag(gammas[n, :])

                np.save(self.tmp_dir + "lambdas_" + str(bno) + "_" + str(i),
                        lambdas_batch)

            else:

                for n in range(sdoc, edoc, 1):
                    alphas[n, :] = np.linalg.solve(self.scov.dot(gammas[n, :, :])
                                                   + eye, tmp[n, :])
                    lambdas_batch[n-sdoc, :, :] = gammas[n, :, :] + spre

                np.save(self.tmp_dir + "lambdas_" + str(bno) + "_" + str(i),
                        lambdas_batch)

            bno += 1
            sdoc += bsize
            edoc += bsize
            if edoc > mus.shape[0]:
                edoc = mus.shape[0]

        return alphas

    def m_step(self, mus, alphas, i, bsize):
        """  M - step: maximize w.r.t params (class mus, shared precision matrix)

            # 1. maximize w.r.t. class mus
            # mu_l = (1/N_l) SUM{n in Class_l} (w_n - alpha_n)

            self.cmus = self.m2l.T.dot(mus - alphas) / self.c_sizes.reshape(-1, 1)

            # 2. maximize w.r.t. shared cov (precision) matrix
            # S = 1/N (SUM_n Lambda_n.inv()) +
            #               (alpha_n - (w_n - mu_l)) (alpha_n - (w_n - mu_l)).T

        """

        # 1 maximizing w.r.t. class means
        self.cmus = self.m2l.T.dot(mus - alphas) / self.c_sizes.reshape(-1, 1)

        # 2 maximizing w.r.t. shared covariance
        self.scov = np.zeros_like(self.scov, dtype=np.float32)

        sdoc = 0
        edoc = bsize
        bno = 0

        tmp = alphas - (mus - self.m2l.dot(self.cmus))
        while sdoc < edoc:

            # print('\r    m-step batch num {:2d}'.format(bno), end=" ")
            lambdas_batch = np.load(self.tmp_dir + "lambdas_" + str(bno) + \
                                    "_" + str(i) + ".npy").astype(np.float32)

            self.scov += (np.linalg.inv(lambdas_batch).sum(axis=0) +
                          tmp[sdoc:edoc, :].T.dot(tmp[sdoc:edoc, :]))

            bno += 1
            sdoc += bsize
            edoc += bsize
            if edoc > mus.shape[0]:
                edoc = mus.shape[0]

        self.scov /= mus.shape[0]

    def train_b(self, data, labels, covs=None, bsize=4096):

        self.train(data, labels, covs=covs, bsize=bsize)

    def train(self, data, labels, covs=None, bsize=4096):
        """ Train using EM

        Args:
            data (np.ndarray): If cov_type is `diag` then, shape is n_docs x [twice dim]
            (every row is twice the dimension of latent variable, where the
             first half is mean, and second half is log std.dev). Otherwise data
             represents only means of shape n_docs x dim.
            labels (np.ndarray): Labels for n_docs. Shape is n_docs
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
            then data represents only means and covs are passed as parameters
            bsize (int): batch size
        """

        if min(labels) == 1:
            labels -= 1

        if self.cov_type == 'diag':
            self.dim = int(data.shape[1] / 2)
            mus = (data[:, :self.dim]).astype(np.float32)
            covs = np.exp(2 * data[:, self.dim:]).astype(np.float32)

        else:
            self.dim = data.shape[1]
            mus = data.astype(np.float32)

        self.compute_cmus_shared_cov(mus, labels)

        if bsize > mus.shape[0]:
            # if batch size is > no. of examples
            bsize = mus.shape[0]

        for i in range(self.trn_iters):

            # print('Iter :{:d}'.format(i))

            alphas = self.e_step(mus, covs, i, bsize)
            # print()
            self.m_step(mus, alphas, i, bsize)
            # print()

            shutil.rmtree(self.tmp_dir)

    def predict_sb(self, data, return_labels=False, covs=None):
        """ Predict log-likelihoods or labels given the test data (means and covs).

        Args:
            data (np.ndarray): If cov_type is `diag` then, shape is n_docs x [twice dim]
            (every row is twice the dimension of latent variable, where the
             first half is mean, and second half is log std.dev). Otherwise data
             represents only means of shape n_docs x dim.
            return_labels (boolean): Returns labels if Trues, else returns log-likelihoods
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
            then data represents only means and covs are passed as parameters

        Returns:
            labels (np.ndarray): Class-conditional LLH or predicted labels for n_docs.
        """

        if self.cov_type == 'diag':
            mus = data[:, :self.dim]
            covs = np.exp(2 * data[:, self.dim:])

            tot_covs = np.zeros(shape=(mus.shape[0], self.dim, self.dim),
                                dtype=np.float32)
            for n in range(mus.shape[0]):
                tot_covs[n, :, :] = self.scov + np.diag(covs[n, :])

        else:
            mus = data
            tot_covs = self.scov + covs

        # inv_tot_covs = np.linalg.inv(tot_covs)
        sgn, log_dets = np.linalg.slogdet(tot_covs)

        const = -0.5 * self.dim * np.log(2 * np.pi)

        if (sgn < 0).any():
            print("WARNING: Det of tot_covs is Negative.")
            sys.exit()

        # class conditional log-likelihoods
        cc_llh = np.zeros(shape=(mus.shape[0], self.cmus.shape[0]),
                          dtype=np.float32)

        for n in range(mus.shape[0]):  # for each doc
            tmp = self.cmus - mus[n, :]
            z = (tmp.reshape(tmp.shape[0], tmp.shape[1], 1) @
                 tmp.reshape(tmp.shape[0], 1, tmp.shape[1]))
            cc_llh[n, :] = const - (0.5 * (log_dets[n] +
                                           (z * np.linalg.inv(
                                               tot_covs[n, :, :])).sum(axis=1).sum(axis=1)))

            # cc_llh[n, :] = -log_dets[n]
            # for i in range(self.cmus.shape[0]):  # given a class, i
            #    cc_llh[n, i] -= (np.outer(tmp[i, :], tmp[i, :]) *
            #                     inv_tot_covs[n, :, :]).sum()

        if self.est_prior:
            if return_labels:
                ret_val = np.argmax(cc_llh + np.log(self.priors).T, axis=1)
            else:
                ret_val = cc_llh + np.log(self.priors).T

        else:
            if return_labels:
                ret_val = np.argmax(cc_llh, axis=1)
            else:
                ret_val = cc_llh

        return ret_val


    def predict_b(self, data, return_labels=False, covs=None, bsize=4096):

        return self.predict(data, return_labels=return_labels, covs=covs, bsize=bsize)

    def predict(self, data, return_labels=False, covs=None, bsize=4096):
        """ Predict log-likelihoods or labels given the test data (means and covs).

        Args:
            data (np.ndarray): If cov_type is `diag` then, shape is n_docs x [twice dim]
            (every row is twice the dimension of latent variable, where the
             first half is mean, and second half is log std.dev). Otherwise data
             represents only means of shape n_docs x dim.
            return_labels (boolean): Returns labels if Trues, else returns log-likelihoods
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
            then data represents only means and covs are passed as parameters
            bsize (int): batch size

        Returns:
            labels (np.ndarray): Class-conditional LLH or predicted labels for n_docs.
        """

        if bsize > data.shape[0]:
            bsize = data.shape[0]

        mus = (data[:, :self.dim]).astype(np.float32)

        # class conditional log-likelihoods
        cc_llh = np.zeros(shape=(mus.shape[0], self.cmus.shape[0]),
                          dtype=np.float32)

        const = -0.5 * self.dim * np.log(2 * np.pi)

        sdoc = 0
        edoc = bsize

        while sdoc < edoc:

            tot_covs = np.zeros(shape=(edoc-sdoc, self.dim, self.dim),
                                dtype=np.float32)

            if self.cov_type == 'diag':
                #  covs = np.exp(2 * data[:, self.dim:])
                for n in range(sdoc, edoc, 1):
                    tot_covs[n-sdoc, :, :] = (self.scov +
                                              np.diag(np.exp(2. * data[n, self.dim:])))

            else:
                tot_covs = self.scov + covs[sdoc:edoc, :, :]

            # inv_tot_covs = np.linalg.inv(tot_covs)
            sgn, log_dets = np.linalg.slogdet(tot_covs)

            if (sgn < 0).any():
                print("WARNING: Det of tot_covs is Negative.")
                sys.exit()

            for n in range(sdoc, edoc, 1):  # for each doc in the batch
                tmp = self.cmus - mus[n, :]
                z = (tmp.reshape(tmp.shape[0], tmp.shape[1], 1) @
                     tmp.reshape(tmp.shape[0], 1, tmp.shape[1]))
                cc_llh[n, :] = const - (0.5 *
                                        (log_dets[n-sdoc] + (z * np.linalg.inv(
                                            tot_covs[n-sdoc, :, :])).sum(axis=1).sum(axis=1)))
            sdoc += bsize
            edoc += bsize
            if edoc > mus.shape[0]:
                edoc = mus.shape[0]

        if self.est_prior:
            if return_labels:
                ret_val = np.argmax(cc_llh + np.log(self.priors).T, axis=1)
            else:
                ret_val = cc_llh + np.log(self.priors).T

        else:
            if return_labels:
                ret_val = np.argmax(cc_llh, axis=1)
            else:
                ret_val = cc_llh

        return ret_val

    def predict_proba(self, data, return_labels=False, covs=None, bsize=4096):
        """ Predict probabilties """

        log_prob = self.predict_b(data)
        pred_probs = np.exp(log_prob.T - lse(log_prob, axis=1)).T
        return pred_probs


def main():
    """ main method """

    np.random.seed(0)
    N, K = 5, 2
    means = np.random.randn(N, K)
    sigmas = np.log(1 / np.sqrt(np.random.gamma(shape=1.5, scale=1.5, size=(N, K))))

    labels = np.random.randint(0, 2, size=(N))
    data = np.concatenate((means, sigmas), axis=1)

    glcu = GLCU(ARGS.trn, cov_type='diag')
    glcu.train(data, labels)

    glcu2 = GLCU(ARGS.trn, cov_type="diag")
    glcu2.train_b(data, labels, bsize=ARGS.bs)

    print('---------')
    print(glcu.scov)
    print(glcu2.scov)
    print(np.allclose(glcu.scov, glcu2.scov))

    print('----------')
    print(glcu.predict(data))
    print(glcu2.predict_b(data))
    print(np.allclose(glcu.predict(data), glcu2.predict_b(data, bsize=ARGS.bs)))


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument('-trn', type=int, default=5)
    PARSER.add_argument('-bs', type=int, default=5)
    ARGS = PARSER.parse_args()
    main()
