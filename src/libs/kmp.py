#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2024. Markus Knauer, Gabriel Quere, João Silvério              #
# All rights reserved.                                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-09-2024                                                             #
# Author: Markus Knauer, Gabriel Quere, Joao Silverio                          #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://gitlab.lrz.de/i23-lectures/2024-masterpraktikum-learning-robotic-skills-from-demonstration  #
################################################################################

"""
Class for the Kernelized Movement Primitives Model (KMP)
"""

__author__ = "Markus Knauer, Maximilian Muehlbauer, Gabriel Quere, Joao Silverio"
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

from typing import List, Tuple, Union

import numpy as np
import scipy.linalg as sp

from src.libs.gmm import GaussianMixtureModel


class Kmp:
    def __init__(
        self,
        gmm_n_components: int = 5,
        N: int = 100,
        l: float = 0.1,
        h: float = 1.0,
        lambda1: float = 0.1,
        lambda2: float = 100,
        alpha: float = 100,
        kernel_function: str = "rbf",
    ):
        """
        A class with basic functionalities of kernelized movement primitives.
        See https://arxiv.org/pdf/1708.08638.pdf

        :param gmm_n_components: Nb of gaussians in the GMM
        :param N: Number of sample points for Gaussian Mixture Regression
        :param l: length scale of the kernel
        :param h: proportionnal kernel scaling factor
        :param lambda1: E(ξ(s*)) = k* (K + λΣ)^-1 μ (21)
        :param lambda2: D(ξ(s*)) = N/λ (k(s*, s*) - k*(K + λΣ)^-1 k*^T) (26)
        :param alpha: KMP covariance prediction proportionnal scaling
        :param kernel_function: "rbf", "matern0", "matern1", "matern2"
        """

        self.gmm_n_components = gmm_n_components
        self.N = N
        self.l = l
        self.h = h
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.nb_via = 0  # start with zero via-points

        self.available_kernels = ["rbf", "matern0", "matern1", "matern2"]
        if kernel_function not in self.available_kernels:
            print("Kernel function not recognized. Options: ", self.available_kernels)
            exit(1)
        else:
            self.kernel_function = kernel_function

    def __repr__(self):
        return f"KMP with:\n\tgmm_n_components = {self.gmm_n_components}\n\tN = {self.N}\n\tl = {self.l}\n\tlambda1 = {self.lambda1}\n\tlambda2 = {self.lambda2}\n\talpha = {self.alpha}"

    def fit(self, data: np.ndarray, d_in: List, d_out: List, gmm_init: Tuple = (None, None, None), x_in: np.ndarray = None):
        """
        Initializes reference trajectory distribution of the KMP.
        (Using GMR as in the original paper, but only needs means and covariances so it can be done with other methods).

        :param data: List of n_feature-dimensional data points. array-like of shape (n_samples, n_features)
        :param d_in: List of input indices for regression
        :param d_out: List of output indices for regression
        :param gmm_init: Tuple of GMM initialization values
        :param x_in: data for regression
        """

        assert isinstance(data, np.ndarray), f"data must be of type np.ndarray, but got {type(data)} instead!"

        means_init, weights_init, precisions_init = gmm_init

        # Train GMM on demonstration data
        gmm = GaussianMixtureModel(
            n_components=self.gmm_n_components,
            covariance_type="full",
            reg_covar=1e-5,
            init_params="kmeans",
            means_init=means_init,
            weights_init=weights_init,
            precisions_init=precisions_init,
        )
        gmm.fit(data)

        # GMR
        # WARN TODO this assumes input / output are ordered
        self.nb_dim_in = np.size(d_in)
        self.nb_dim_out = np.size(d_out)

        if isinstance(self.l, (np.ndarray, list)):
            self.l = np.array(self.l)
        else:
            self.l = self.l * np.ones(self.nb_dim_in)

        if x_in is None:
            if self.nb_dim_in == 1:
                x_in = np.linspace(0, 1, self.N)
            else:
                x_in = np.zeros((self.N, self.nb_dim_in))
                for i in range(0, self.N):
                    state = np.random.choice(self.gmm_n_components, p=gmm.weights_)
                    x_in[i, :] = np.random.multivariate_normal(
                        gmm.means_[state, 0 : self.nb_dim_in], gmm.covariances_[state, 0 : self.nb_dim_in, 0 : self.nb_dim_in]
                    )

        assert isinstance(x_in, np.ndarray), f"x_in must be of type np.ndarray, but got {type(x_in)} instead!"
        assert (
            x_in.shape[0] == self.N
        ), f"The first dimension of the input must match the number of sample points! But got {x_in.shape[0]} as first dimension of the input and {self.N} number of sample points."
        if len(x_in.shape) >= 2:
            assert (
                x_in.shape[1] == self.nb_dim_in
            ), f"The second dimension of the input must match the given input size! But got {x_in.shape[1]} as second dimension of the input and {self.nb_dim_in} as given input size."
        mu, sigma, _ = gmm.gaussian_mixture_regression(x_in, d_in, d_out, self.N)

        # Block-ize GMR output
        mu_block = mu.reshape(-1, 1)  # mean
        pntr_sigma = []
        for i in range(0, sigma.shape[0]):  # covariance
            pntr_sigma.append(sigma[i, :, :])
        sigma_block = sp.block_diag(*pntr_sigma)

        # need a 2d array because input might be multi-dim
        self.x_in = x_in.reshape(-1, self.nb_dim_in)
        # this is kept for analysis, could be discarded otherwise
        self.model = gmm
        self.mu = mu
        self.sigma = sigma
        self.mu_block = mu_block
        self.sigma_block = sigma_block
        self.update_K()

        self.mean_trace = None

    def rbf_kernel(self, x1, x2, l, h=1.0):
        """Implementation of the rbf kernel."""

        diff = np.repeat(x1[:, np.newaxis, :], x2.shape[0], axis=1) - np.repeat(x2[np.newaxis, :, :], x1.shape[0], axis=0)
        squared_dist = np.einsum("...i,ij,...j->...", diff, np.diag(l ** (-2)), diff)
        return h**2 * np.exp(-squared_dist / 2)

    def matern_kernel_p0(self, x1, x2, l, h=1.0):
        """Implementation of the Matern kernel with p=0."""

        diff = np.repeat(x1[:, np.newaxis, :], x2.shape[0], axis=1) - np.repeat(x2[np.newaxis, :, :], x1.shape[0], axis=0)
        dist = np.sqrt(np.einsum("...i,ij,...j->...", diff, np.diag(l ** (-2)), diff))
        return h**2 * np.exp(-dist)

    def matern_kernel_p1(self, x1, x2, l, h=1.0):
        """Implementation of the Matern kernel with p=1."""

        diff = np.repeat(x1[:, np.newaxis, :], x2.shape[0], axis=1) - np.repeat(x2[np.newaxis, :, :], x1.shape[0], axis=0)
        dist = np.sqrt(3 * np.einsum("...i,ij,...j->...", diff, np.diag(l ** (-2)), diff))
        return h**2 * (1 + dist) * np.exp(-dist)

    def matern_kernel_p2(self, x1, x2, l, h=1.0):
        """Implementation of the Matern kernel with p=2."""

        diff = np.repeat(x1[:, np.newaxis, :], x2.shape[0], axis=1) - np.repeat(x2[np.newaxis, :, :], x1.shape[0], axis=0)
        squared_dist = 5 * np.einsum("...i,ij,...j->...", diff, np.diag(l ** (-2)), diff)
        dist = np.sqrt(squared_dist)
        return h**2 * (1 + dist + squared_dist / 3) * np.exp(-dist)

    def kernel_matrix(self, x1, x2, l, h):
        """Computation of the kernel matrix for two inputs."""

        I_O = np.eye(self.nb_dim_out)
        if self.kernel_function is self.available_kernels[0]:
            return np.kron(self.rbf_kernel(x1, x2, l, h), I_O)
        elif self.kernel_function is self.available_kernels[1]:
            return np.kron(self.matern_kernel_p0(x1, x2, l, h), I_O)
        elif self.kernel_function is self.available_kernels[2]:
            return np.kron(self.matern_kernel_p1(x1, x2, l, h), I_O)
        else:
            return np.kron(self.matern_kernel_p2(x1, x2, l, h), I_O)

    def update_K(self):
        """Computes kernel matrix for all outputs."""

        self.K = self.kernel_matrix(self.x_in, self.x_in, self.l, self.h)
        self.invK = np.linalg.inv(self.K + self.lambda1 * self.sigma_block)
        # Used for the covariance prediction, difference is lambda2; might be relaxed if we assume lambda2=lambda1
        self.invK2 = np.linalg.inv(self.K + self.lambda2 * self.sigma_block)
        # term used for the computation of epistemic uncertainty
        self.invK_epi = np.linalg.inv(
            self.K + 1e-8 * np.eye((self.N + self.nb_via) * self.nb_dim_out, (self.N + self.nb_via) * self.nb_dim_out)
        )

    def update_inputs(self, x_test):
        """Computes K_s and K_ss, the kernel matrices that depend on test inputs."""

        self.x_test = x_test
        self.K_s = self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), self.x_in, self.l, self.h)
        self.K_ss = self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), x_test.reshape(-1, self.nb_dim_in), self.l, self.h)

    def mean(self):
        """KMP mean prediction."""

        self.mu_out = (self.K_s @ self.invK @ self.mu_block).reshape((-1, self.nb_dim_out))
        return self.mu_out

    def epistemic(self):
        """KMP epistemic prediction."""

        self.epi_uncertainty = self.K_ss - self.K_s @ self.invK_epi @ self.K_s.T
        return self.epi_uncertainty

    def aleatoric(self):
        """KMP aleatoric prediction."""

        self.al_uncertainty = (
            self.K_s
            @ np.linalg.inv(
                self.K @ np.linalg.inv(self.lambda2 * self.sigma_block) @ self.K
                + self.K
                + 1e-10 * np.eye((self.N + self.nb_via) * self.nb_dim_out)
            )
            @ self.K_s.T
        )
        return self.al_uncertainty

    def cov(self):
        """KMP covariance prediction."""

        self.sigma_out = self.alpha * (self.K_ss - self.K_s @ self.invK2 @ self.K_s.T)
        return self.sigma_out

    def diag_blocks(self, orig_matrix):
        """
        Call after cov() since sigma is a full cov matrix!

        This function extracts the diagonal blocks, to get the matrix
        of the output for each input point.
        """
        assert "sigma_out" in dir(self), "Error: [diag_blocks] called before [cov]"

        self.x_test = self.x_test.reshape(-1, self.nb_dim_in)
        diag_matrix = np.zeros((len(self.x_test), self.nb_dim_out, self.nb_dim_out))
        j = 0
        for i in range(0, orig_matrix.shape[0], self.nb_dim_out):
            diag_matrix[j, :, :] = orig_matrix[i : i + self.nb_dim_out][:, i : i + self.nb_dim_out]
            j += 1
        return diag_matrix

    def predict(self, x_test):
        """Computes mean and covariance predictions for test input."""
        self.update_inputs(x_test)
        return self.mean(), self.cov()

    def add_viapoints(
        self,
        input_via: Union[float, list, np.ndarray],
        output_via: [float, list, np.ndarray],
        gamma: int = 1e-8,
        replace: bool = False,
    ):
        """Adds via-points to the KMP.
        :param input_via: Position where the via-point will be added (e. g. time) this can either be a list or only one value
        :param output_via: The via-point (e.g. in x,y,z), this can also be a list or only one value
        """
        if not isinstance(input_via, np.ndarray):
            if isinstance(input_via, float):
                input_via = np.array([input_via])
            elif isinstance(input_via, list):
                pass
            else:
                raise Exception(f"only np.ndarray or float values are allowed, not {type(input_via)}")
        if not isinstance(output_via, np.ndarray):
            if isinstance(output_via, float):
                output_via = np.array([output_via])
            elif isinstance(output_via, list):
                pass
            else:
                raise Exception(f"only np.ndarray or float values are allowd, not {type(output_via)}")
        assert len(input_via) == len(output_via)

        I_O = np.eye(self.nb_dim_out, self.nb_dim_out)
        precision = float(gamma) * I_O

        ### Recompute K, k*, k**
        for i in range(len(input_via)):
            via_point = [input_via[i], output_via[i], float(gamma) * I_O]

            if replace:
                idx = np.searchsorted(self.x_in.reshape(-1), via_point[0])[0]
                self.mu_block[int(idx * self.nb_dim_out) : int(idx * self.nb_dim_out + self.nb_dim_out)] = via_point[
                    1
                ].reshape((-1, 1))
                self.sigma[idx] = via_point[2].reshape(self.nb_dim_out, -1)
            else:
                self.x_in = np.append(self.x_in, via_point[0])  # crucial step 1)
                self.mu_block = np.append(self.mu_block, via_point[1].T)  # crucial step 2)
                self.sigma = np.append(self.sigma, precision.reshape(1, self.nb_dim_out, -1), axis=0)  # crucial step 3)
                self.nb_via += 1
                # self.N = self.N + 1
        self.x_in = self.x_in.reshape(-1, self.nb_dim_in)

        # Blockize GMR output + the new covariance
        pntr_sigma = []
        for i in range(0, self.sigma.shape[0]):
            pntr_sigma.append(self.sigma[i, :, :])
        self.sigma_block = sp.block_diag(*pntr_sigma)
        self.update_K()
        self.update_inputs(self.x_test)
