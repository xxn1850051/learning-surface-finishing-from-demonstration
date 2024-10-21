#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2024. Markus Knauer, Gabriel Quere, Joao Silverio              #
# All rights reserved.                                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-09-2024                                                             #
# Author: Markus Knauer, Gabriel Quere, Joao Silverio                          #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://gitlab.lrz.de/i23-lectures/2024-masterpraktikum-learning-robotic-skills-from-demonstration  #
################################################################################

"""
Class for the Gaussian Mixture Model (GMM)
"""

__author__ = "Markus Knauer, Gabriel Quere, Joao Silverio"
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel(GaussianMixture):
    def __init__(self, **kwargs):
        """
        Inherit from from sklearn.mixture.GaussianMixture and add gaussian_mixture_regression
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        """
        super().__init__(**kwargs)

    def approximate_single_gaussian(self, weights) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate a single gaussian.

        :param weights: The weights to use.
        :return: Mean & covariance.
        """

        mean = sum([self.means_[i, :] * weights[i] for i in range(self.n_components)])
        diffs = mean.reshape(1, -1) - self.means_
        covariance = sum([(self.covariances_[i] + diffs[i] @ diffs[i].T) * weights[i] for i in range(self.n_components)])
        return mean, covariance

    def gaussian_margin(self, d_marg: List) -> "GaussianMixtureModel":
        """
        Perform marginalization and return .

        :param d_marg: Indices for marginalization.
        :return: Mean and covariances.
        """

        means = []
        covariances = []
        for i in range(0, self.n_components):
            means.append(self.means_[i, np.ix_(d_marg)].reshape(1, -1))
            covariances.append(self.covariances_[i][np.ix_(d_marg, d_marg)])

        new_gmm = GaussianMixtureModel(n_components=self.n_components)
        new_gmm.means_ = np.vstack(means)
        new_gmm.covariances_ = np.array(covariances)
        new_gmm.weights_ = self.weights_
        new_gmm.converged_ = self.converged_
        new_gmm.lower_bound_ = self.lower_bound_
        new_gmm.n_iter_ = self.n_iter_
        new_gmm.precisions_ = np.linalg.inv(new_gmm.covariances_)
        new_gmm.precisions_cholesky_ = np.linalg.cholesky(new_gmm.precisions_).transpose((0, 2, 1))
        return new_gmm

    def gaussian_conditioning(self, index: int, x_in: np.array, d_in: Optional[List] = None, d_out: Optional[List] = None):
        """
        :param mean: Means of model
        :param cov: Covariance of model
        :param x_in: input data
        :param d_in: x_in indices
        :param d_out: indexes to condition onto
        """
        if d_in is None:
            d_in = range(0, 1)
        if d_out is None:
            d_out = range(1, 2)

        mean = self.means_[index]
        cov = self.covariances_[index]
        nb_dim_in = np.size(d_in)
        nb_dim_out = np.size(d_out)

        mu_ii = mean[d_in].reshape(nb_dim_in, -1)
        mu_oo = mean[d_out].reshape(nb_dim_out, -1)
        cov_ii = cov[np.ix_(d_in, d_in)]
        prec_ii = np.linalg.inv(cov_ii)
        cov_io = cov[np.ix_(d_in, d_out)]
        cov_oi = cov_io.T
        cov_oo = cov[np.ix_(d_out, d_out)]

        # conditional distribution
        mu_cond = mu_oo + cov_oi @ prec_ii @ (x_in.T - mu_ii)
        mu_cond = mu_cond.T  # features as columns
        cov_cond = cov_oo - cov_oi @ prec_ii @ cov_io

        return mu_cond, cov_cond

    def gaussian_mixture_regression(
        self,
        x_in: np.array,
        d_in: Optional[List] = None,
        d_out: Optional[List] = None,
        N: Optional[int] = None,
        single_gaussian: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        :param x_in: input data
        :param d_in: x_in indices
        :param d_out: indexes to condition onto
        :param N: len(x_in) or shorter ? Is that even required ?
        :param single_gaussian: if True use moment matching (one Gaussian), else return whole GMM
        """
        # todo: derive d_out from d_in if d_out is None
        if d_in is None and d_out is None:
            d_in = [0]
            d_out = [1]
        else:
            assert d_in is not None and d_out is not None

        assert len(x_in.shape) <= 2, f"The input must be one or two dimensional!, got dimensionality of {len(x_in.shape)}"
        if len(x_in.shape) == 2:
            assert x_in.shape[1] == len(
                d_in
            ), f"The input must have the same size as the requested list of dimensions! But got input size of {x_in.shape[1]} and requested dimension of {len(d_in)}"
        else:  # For 1D Input
            assert x_in.ndim == len(
                d_in
            ), f"The input must have the same size as the requested list of dimensions! But got input size of {x_in.ndim} and requested dimension of {len(d_in)}"

        if N is None:
            if len(x_in.shape) == 2:
                N = np.shape(x_in)[0]
            else:
                N = np.shape(x_in)

        nb_dim_in = np.size(d_in)
        nb_dim_out = np.size(d_out)

        assert (
            self.means_.shape[1] == nb_dim_out + nb_dim_in
        ), f"All used input and output dimensions have to be used, in the definition of what is in and output! Here means shape is {self.means_.shape[1]} and sum of in- and output dimensions is {nb_dim_in+nb_dim_out}"

        # initialize variables to store conditional distributions
        mu_cond = np.zeros((N, len(d_out), self.n_components))
        sigma_cond = np.zeros(
            (len(d_out), len(d_out), self.n_components)
        )  # doesn't need N because the covariance of the conditional is not input-dep.

        # to store moment matching approximation
        mu = np.zeros((N, len(d_out)))
        sigma = np.zeros((N, len(d_out), len(d_out)))

        h = np.zeros((N, self.n_components))

        for i in range(0, self.n_components):
            # marginal distribution of the input variable
            mu_ii = self.means_[i, np.ix_(d_in)].reshape(nb_dim_in, -1)
            cov_ii = self.covariances_[i][np.ix_(d_in, d_in)]

            # conditional distribution for each Gaussian
            mu_cond[:, :, i], sigma_cond[:, :, i] = self.gaussian_conditioning(i, x_in, d_in, d_out)

            # prior update
            if len(x_in.shape) > 1 and x_in.shape[1] == 1:
                h[:, i] = self.weights_[i] * multivariate_normal.pdf(x_in[:, 0], mean=mu_ii.flatten(), cov=cov_ii)
            else:
                h[:, i] = self.weights_[i] * multivariate_normal.pdf(x_in, mean=mu_ii.flatten(), cov=cov_ii)

        h = h / np.sum(h, axis=1)[:, None]  # priors must sum to 1

        if single_gaussian:
            # moment matching to approximate multiple gaussians with just one
            for i in range(N):
                mu[i, :] = mu_cond[i, :, :] @ h[i, :]
                sigma_tmp = np.zeros((nb_dim_out, nb_dim_out))
                for n in range(self.n_components):
                    sigma_tmp += h[i, n] * (sigma_cond[:, :, n] + np.outer(mu_cond[i, :, n], mu_cond[i, :, n]))
                sigma[i, :, :] = sigma_tmp - np.outer(mu[i, :], mu[i, :])
            return mu, sigma, 1.0
        else:
            return mu_cond, sigma_cond, h
