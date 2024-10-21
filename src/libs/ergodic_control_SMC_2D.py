#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2024. Stefan Schneyer                                          #
# All rights reserved.                                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 18-09-2024                                                             #
# Author: Stefan Schneyer                                                      #
# E-mail: stefan.schneyer@dlr.de                                               #
# Original src from: https://gitlab.idiap.ch/rli/robotics-codes-from-scratch  #
# Written by Philip Abbet <philip.abbet@idiap.ch> and Sylvain Calinon <https://calinon.ch> #
# Website: https://gitlab.lrz.de/i23-lectures/2024-masterpraktikum-learning-robotic-skills-from-demonstration  #
################################################################################

"""
    2D ergodic control formulated as Spectral Multiscale Coverage (SMC) objective,
    with a spatial distribution described as a mixture of Gaussians.
"""

__author__ = "Stefan Schneyer"
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Stefan Schneyer"
__email__ = "stefan.schneyer@dlr.de"

# from math import exp
import matplotlib.pyplot as plt
import numpy as np


# Helper functions
# ===============================
def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
    # Base case: A Hadamard matrix of size 1 is just [[1]].
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2.
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    # Combine the four sub-matrices to form a Hadamard matrix of size n.
    h = np.empty((n, n), dtype=int)
    h[:half_size, :half_size] = h_half
    h[half_size:, :half_size] = h_half
    h[:half_size:, half_size:] = h_half
    h[half_size:, half_size:] = -h_half

    return h


class ErgodicControlSMC2D:

    def __init__(self):
        # Parameters
        # ===============================
        self.nbData = 500  # Number of datapoints
        self.nbFct = 8  # Number of basis functions along x and y
        self.nbVar = 2  # Dimension of datapoints
        # Number of Gaussians to represent the spatial distribution
        self.nbGaussian = 2
        self.sp = (self.nbVar + 1) / 2  # Sobolev norm parameter
        self.dt = 1e-2  # Time step
        # Domain limit for each dimension (considered to be 1
        # for each dimension in this implementation)
        self.xlim = [0, 1]
        self.L = (self.xlim[1] - self.xlim[0]) * 2  # Size of [-self.xlim(2),self.xlim(2)]
        self.om = 2 * np.pi / self.L
        self.u_max = 3e0  # Maximum speed allowed
        self.u_norm_reg = 1e-3  # Regularizer to avoid numerical issues when speed is close to zero
        self.nbRes = 100

        # Desired spatial distribution represented as a mixture of Gaussians (GMM)
        # gaussian centers
        self.mu = np.zeros((self.nbVar, self.nbGaussian))
        self.mu[:, 0] = np.array([0.5, 0.7])
        self.mu[:, 1] = np.array([0.6, 0.3])
        # Gaussian covariances
        # Direction vectors for constructing the covariance matrix using
        # outer product of a vector with itself then the principal direction
        # of covariance matrix becomes the given vector and its orthogonal
        # complement
        sigma1_v = [0.3, 0.1]
        sigma2_v = [0.1, 0.2]
        # scaling terms
        sigma1_scale = 5e-1
        sigma2_scale = 3e-1
        # regularization terms
        sigma1_regularization = np.eye(self.nbVar) * 5e-3
        sigma2_regularization = np.eye(self.nbVar) * 1e-2
        # cov. matrices
        self.sigma = np.zeros((self.nbVar, self.nbVar, self.nbGaussian))
        # construct the cov. matrix using the outer product
        self.sigma[:, :, 0] = np.vstack(sigma1_v) @ np.vstack(sigma1_v).T * sigma1_scale + sigma1_regularization
        self.sigma[:, :, 1] = np.vstack(sigma2_v) @ np.vstack(sigma2_v).T * sigma2_scale + sigma2_regularization
        self.prior = np.ones(self.nbGaussian) / self.nbGaussian

        self.compute_w_hat()
        self.fourier_basis_functions()

        # Desired spatial distribution
        self.g = self.w_hat.T @ self.phim

        self.wt = np.zeros(self.nbFct**self.nbVar)
        self.r_x = np.zeros((self.nbVar, self.nbData))

    def plot(self):
        # Plot
        # ===============================
        fig, ax = plt.subplots(1, 3, figsize=(16, 8))
        G = np.reshape(self.g, [self.nbRes, self.nbRes])  # original distribution
        G = np.where(G > 0, G, 0)
        # G = np.reshape(r_g[:, -1], [self.nbRes, self.nbRes])  # reconstructed spatial distribution
        # x
        X = np.squeeze(self.xm[0, :, :])
        Y = np.squeeze(self.xm[1, :, :])
        ax[0].contourf(X, Y, G, cmap="gray_r")
        ax[0].plot(self.r_x[0, :], self.r_x[1, :], linestyle="-", color="black")
        ax[0].plot(self.r_x[0, 0], self.r_x[1, 0], marker=".", color="black", markersize=10)
        ax[0].set_aspect("equal", "box")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # w_hat
        ax[1].set_title(r"Desired Fourier coefficients $\hat{w}$")
        ax[1].imshow(np.reshape(self.w_hat, [self.nbFct, self.nbFct]).T, cmap="gray_r")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # w
        ax[2].set_title(r"Reproduced Fourier coefficients $w$")
        ax[2].imshow(np.reshape(self.wt / self.nbData, [self.nbFct, self.nbFct]).T, cmap="gray_r")
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        plt.show()

    def run_from_init_pos(self, x0):
        # Ergodic control
        # ===============================
        traj = []
        x = np.array(x0)  # Initial position
        r_g = np.zeros((self.nbRes**self.nbVar, self.nbData))
        r_w = np.zeros((self.nbFct**self.nbVar, self.nbData))
        r_e = np.zeros((self.nbData))
        for t in range(self.nbData):
            w, x = self.ergodic_step(t, x)
            # Log data
            self.r_x[:, t] = x
            # Reconstructed spatial distribution (for visualization)
            r_g[:, t] = self.phim.T @ w
            # Fourier coefficients along trajectory (for visualization)
            r_w[:, t] = w
            # Reconstruction error evaluation
            r_e[t] = np.sum((w - self.w_hat) ** 2 * self.Lambda)
            traj.append(x)
        return traj

    def ergodic_step(self, t, x):
        # Fourier basis functions and derivatives for each dimension
        # (only cosine part on [0,L/2] is computed since the signal
        # is even and real by construction)
        angle = x[:, np.newaxis] * self.rg * self.om
        phi1 = np.cos(angle) / self.L
        dphi1 = -np.sin(angle) * np.tile(self.rg * self.om, (self.nbVar, 1)) / self.L
        # Gradient of basis functions
        phix = phi1[0, self.xx - 1].flatten()
        phiy = phi1[1, self.yy - 1].flatten()
        dphix = dphi1[0, self.xx - 1].flatten()
        dphiy = dphi1[1, self.yy - 1].flatten()
        dphi = np.vstack([[dphix * phiy], [phix * dphiy]]).T
        # w are the Fourier series coefficients along trajectory
        self.wt = self.wt + (phix * phiy).T
        w = self.wt / (t + 1)
        # Controller with constrained velocity norm
        u = -dphi.T @ (self.Lambda * (w - self.w_hat))
        u = u * self.u_max / (np.linalg.norm(u) + self.u_norm_reg)  # Velocity command
        x = x + (u * self.dt)  # Update of position
        return w, x

    def fourier_basis_functions(self):
        # Fourier basis functions (for a discretized map)
        # ===============================
        self.xm1d = np.linspace(self.xlim[0], self.xlim[1], self.nbRes)  # Spatial range for 1D
        self.xm = np.zeros((self.nbGaussian, self.nbRes, self.nbRes))  # Spatial range
        self.xm[0, :, :], self.xm[1, :, :] = np.meshgrid(self.xm1d, self.xm1d)
        # Mind the flatten() !!!
        arg1 = self.KX[0, :, :].flatten().T[:, np.newaxis] @ self.xm[0, :, :].flatten()[:, np.newaxis].T * self.om
        arg2 = self.KX[1, :, :].flatten().T[:, np.newaxis] @ self.xm[1, :, :].flatten()[:, np.newaxis].T * self.om
        self.phim = np.cos(arg1) * np.cos(arg2) * 2 ** (self.nbVar)  # Fourrier basis functions
        # Some weird +1, -1 due to 0 index!!!
        self.xx, self.yy = np.meshgrid(np.arange(1, self.nbFct + 1), np.arange(1, self.nbFct + 1))
        hk = np.concatenate(([1], 2 * np.ones(self.nbFct)))
        HK = hk[self.xx.flatten() - 1] * hk[self.yy.flatten() - 1]
        self.phim = self.phim * np.tile(HK, (self.nbRes**self.nbVar, 1)).T

    def compute_w_hat(self):
        # Compute Fourier series coefficients self.w_hat of desired spatial distribution
        # ===============================
        self.rg = np.arange(0, self.nbFct, dtype=float)
        self.KX = np.zeros((self.nbVar, self.nbFct, self.nbFct))
        self.KX[0, :, :], self.KX[1, :, :] = np.meshgrid(self.rg, self.rg)
        # Mind the flatten() !!!
        # Weighting vector (Eq.(16))
        self.Lambda = np.array(self.KX[0, :].flatten() ** 2 + self.KX[1, :].flatten() ** 2 + 1).T ** (-self.sp)
        # Explicit description of self.w_hat by exploiting the Fourier transform
        # properties of Gaussians (optimized version by exploiting symmetries)
        op = hadamard_matrix(2 ** (self.nbVar - 1))
        op = np.array(op)
        kk = self.KX.reshape(self.nbVar, self.nbFct**2) * self.om
        # compute self.w_hat
        self.w_hat = np.zeros(self.nbFct**self.nbVar)
        for j in range(self.nbGaussian):
            for n in range(op.shape[1]):
                MuTmp = np.diag(op[:, n]) @ self.mu[:, j]
                SigmaTmp = np.diag(op[:, n]) @ self.sigma[:, :, j] @ np.diag(op[:, n]).T
                cos_term = np.cos(kk.T @ MuTmp)
                exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
                # Eq.(22) where D=1
                self.w_hat = self.w_hat + self.prior[j] * cos_term * exp_term
        self.w_hat = self.w_hat / (self.L**self.nbVar) / (op.shape[1])
