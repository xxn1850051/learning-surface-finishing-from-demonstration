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
# Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>                            #
# Website: https://gitlab.lrz.de/i23-lectures/2024-masterpraktikum-learning-robotic-skills-from-demonstration  #
################################################################################

"""
    2D ergodic control formulated as Heat Equation Driven Area Coverage (HEDAC) objective,
	with a spatial distribution described as a mixture of Gaussians.
"""

__author__ = "Stefan Schneyer"
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Stefan Schneyer"
__email__ = "stefan.schneyer@dlr.de"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Helper class
# ===============================
class SecondOrderAgent:
    """
    A point mass agent with second order dynamics.
    """

    def __init__(
        self,
        x,
        nbDataPoints,
        max_dx=1,
        max_ddx=0.2,
    ):
        self.x = np.array(x)  # position
        # determine which dimesnion we are in from given position
        self.nbVarX = len(x)
        self.dx = np.zeros(self.nbVarX)  # velocity

        self.t = 0  # time
        self.dt = 1  # time step
        self.nbDatapoints = nbDataPoints

        self.max_dx = max_dx
        self.max_ddx = max_ddx

        # we will store the actual and desired position
        # of the agent over the timesteps
        self.x_arr = np.zeros((self.nbDatapoints, self.nbVarX))
        self.des_x_arr = np.zeros((self.nbDatapoints, self.nbVarX))

    def update(self, gradient):
        """
        set the acceleration of the agent to clamped gradient
        compute the position at t+1 based on clamped acceleration
        and velocity
        """
        ddx = gradient  # we use gradient of the potential field as acceleration
        # clamp acceleration if needed
        if np.linalg.norm(ddx) > self.max_ddx:
            ddx = self.max_ddx * ddx / np.linalg.norm(ddx)

        # import pdb; pdb.set_trace()
        self.x = self.x + self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        if self.t < self.x_arr.shape[0]:
            self.x_arr[self.t] = np.copy(self.x)
        self.t += 1

        self.dx += self.dt * ddx  # compute the velocity
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)


# Helper functions for HEDAC
# ===============================
def rbf(mean, x, eps):
    """
    Radial basis function w/ Gaussian Kernel
    """
    d = x - mean  # radial distance
    l2_norm_squared = np.dot(d, d)
    # eps is the shape parameter that can be interpreted as the inverse of the radius
    return np.exp(-eps * l2_norm_squared)


def normalize_mat(mat):
    return mat / (np.sum(mat) + 1e-10)


def clamp_kernel_1d(x, low_lim, high_lim, kernel_size):
    """
    A function to calculate the start and end indices
    of the kernel around the agent that is inside the grid
    i.e. clamp the kernel by the grid boundaries
    """
    start_kernel = low_lim
    start_grid = x - (kernel_size // 2)
    num_kernel = kernel_size
    # bound the agent to be inside the grid
    if x <= -(kernel_size // 2):
        x = -(kernel_size // 2) + 1
    elif x >= high_lim + (kernel_size // 2):
        x = high_lim + (kernel_size // 2) - 1

    # if agent kernel around the agent is outside the grid,
    # clamp the kernel by the grid boundaries
    if start_grid < low_lim:
        start_kernel = kernel_size // 2 - x - 1
        num_kernel = kernel_size - start_kernel - 1
        start_grid = low_lim
    elif start_grid + kernel_size >= high_lim:
        num_kernel -= x - (high_lim - num_kernel // 2 - 1)
    if num_kernel > low_lim:
        grid_indices = slice(start_grid, start_grid + num_kernel)

    return grid_indices, start_kernel, num_kernel


def agent_block(min_val, agent_radius):
    """
    A matrix representing the shape of an agent (e.g, RBF with Gaussian kernel).
    min_val is the upper bound on the minimum value of the agent block.
    """
    nbVarX = 2  # number of dimensions of space

    eps = 1.0 / agent_radius  # shape parameter of the RBF
    l2_sqrd = -np.log(min_val) / eps  # squared maximum distance from the center of the agent block
    l2_sqrd_single = l2_sqrd / nbVarX  # maximum squared distance on a single axis since sum of all axes equal to l2_sqrd
    l2_single = np.sqrt(l2_sqrd_single)  # maximum distance on a single axis
    # round to the nearest larger integer
    if l2_single.is_integer():
        l2_upper = int(l2_single)
    else:
        l2_upper = int(l2_single) + 1
    # agent block is symmetric about the center
    num_rows = l2_upper * 2 + 1
    num_cols = num_rows
    block = np.zeros((num_rows, num_cols))
    center = np.array([num_rows // 2, num_cols // 2])
    for i in range(num_rows):
        for j in range(num_cols):
            block[i, j] = rbf(np.array([j, i]), center, eps)
    # we hope this value is close to zero
    print(f"Minimum element of the block: {np.min(block)}" + " values smaller than this assumed as zero")
    return block


def offset(mat, i, j):
    """
    offset a 2D matrix by i, j
    """
    rows, cols = mat.shape
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]


def border_interpolate(x, length, border_type):
    """
    Helper function to interpolate border values based on the border type
    (gives the functionality of cv2.borderInterpolate function)
    """
    if border_type == "reflect101":
        if x < 0:
            return -x
        elif x >= length:
            return 2 * length - x - 2
    return x


def bilinear_interpolation(grid, pos):
    """
    Linear interpolating function on a 2-D grid
    """
    x, y = pos.astype(int)
    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1], "reflect101")
    x1 = border_interpolate(x + 1, grid.shape[1], "reflect101")
    y0 = border_interpolate(y, grid.shape[0], "reflect101")
    y1 = border_interpolate(y + 1, grid.shape[0], "reflect101")
    # Distance from lower integers
    xd = pos[0] - x0
    yd = pos[1] - y0
    # Interpolate on x-axis
    c01 = grid[y0, x0] * (1 - xd) + grid[y0, x1] * xd
    c11 = grid[y1, x0] * (1 - xd) + grid[y1, x1] * xd
    # Interpolate on y-axis
    c = c01 * (1 - yd) + c11 * yd
    return c


# Helper functions borrowed from SMC example given in
# demo_ergodicControl_2D_01.py for using the same
# target distribution and comparing the results
# of SMC and HEDAC
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


def get_GMM(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Gaussian centers
    Mu1 = [0.5, 0.7]
    Mu2 = [0.6, 0.3]
    # Gaussian covariances
    # direction vectors for constructing the covariance matrix using
    # outer product of a vector with itself then the principal direction
    # of covariance matrix becomes the given vector and its orthogonal
    # complement
    Sigma1_v = [0.3, 0.1]
    Sigma2_v = [0.1, 0.2]
    # scale
    Sigma1_scale = 5e-1
    Sigma2_scale = 3e-1
    # regularization
    Sigma1_regularization = np.eye(param.nbVarX) * 5e-3
    Sigma2_regularization = np.eye(param.nbVarX) * 1e-2
    # GMM Gaussian Mixture Model

    # Gaussian centers
    Mu = np.zeros((param.nbVarX, param.nbGaussian))
    Mu[:, 0] = np.array(Mu1)
    Mu[:, 1] = np.array(Mu2)
    # covariance matrices
    Sigma = np.zeros((param.nbVarX, param.nbVarX, param.nbGaussian))
    # construct the covariance matrix using the outer product
    Sigma[:, :, 0] = np.vstack(Sigma1_v) @ np.vstack(Sigma1_v).T * Sigma1_scale + Sigma1_regularization
    Sigma[:, :, 1] = np.vstack(Sigma2_v) @ np.vstack(Sigma2_v).T * Sigma2_scale + Sigma2_regularization
    # mixing. coefficients Priors (summing to one)
    Alpha = np.ones(param.nbGaussian) / param.nbGaussian
    return Mu, Sigma, Alpha


def get_fixed_GMM(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Gaussian centers
    Mu1 = [0.5, 0.7]
    Mu2 = [0.6, 0.3]
    # Gaussian covariances
    # direction vectors for constructing the covariance matrix using
    # outer product of a vector with itself then the principal direction
    # of covariance matrix becomes the given vector and its orthogonal
    # complement
    Sigma1_v = [0.3, 0.1]
    Sigma2_v = [0.1, 0.2]

    Sigma1_scale = 5e-1
    Sigma2_scale = 3e-1

    Sigma1_regularization = np.eye(param.nbVarX) * 5e-3
    Sigma2_regularization = np.eye(param.nbVarX) * 1e-2

    # GMM Gaussian Mixture Model
    Mu = np.zeros((param.nbVarX, param.nbGaussian))
    Mu[:, 0] = np.array(Mu1)
    Mu[:, 1] = np.array(Mu2)
    # covariance matrices
    Sigma = np.zeros((param.nbVarX, param.nbVarX, param.nbGaussian))
    # construct the covariance matrix using the outer product
    Sigma[:, :, 0] = np.vstack(Sigma1_v) @ np.vstack(Sigma1_v).T * Sigma1_scale + Sigma1_regularization
    Sigma[:, :, 1] = np.vstack(Sigma2_v) @ np.vstack(Sigma2_v).T * Sigma2_scale + Sigma2_regularization
    # mixing coefficients priors (summing to one)
    Alpha = np.ones(param.nbGaussian) / param.nbGaussian
    return Mu, Sigma, Alpha


def discrete_gmm(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Discretize given GMM using Fourier basis functions
    rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVarX, param.nbFct, param.nbFct))
    KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
    # Mind the flatten() !!!

    # Explicit description of w_hat by exploiting the Fourier transform
    # properties of Gaussians (optimized version by exploiting symmetries)
    op = hadamard_matrix(2 ** (param.nbVarX - 1))
    op = np.array(op)
    # check the reshaping dimension !!!
    kk = KX.reshape(param.nbVarX, param.nbFct**2) * param.omega

    # Compute fourier basis function weights w_hat for the target distribution given by GMM
    w_hat = np.zeros(param.nbFct**param.nbVarX)
    for j in range(param.nbGaussian):
        for n in range(op.shape[1]):
            MuTmp = np.diag(op[:, n]) @ param.Mu[:, j]
            SigmaTmp = np.diag(op[:, n]) @ param.Sigma[:, :, j] @ np.diag(op[:, n]).T
            cos_term = np.cos(kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
            # Eq.(22) where D=1
            w_hat = w_hat + param.Alpha[j] * cos_term * exp_term
    w_hat = w_hat / (param.L**param.nbVarX) / (op.shape[1])

    # Fourier basis functions (for a discretized map)
    xm1d = np.linspace(param.xlim[0], param.xlim[1], param.nbRes)  # Spatial range
    xm = np.zeros((param.nbGaussian, param.nbRes, param.nbRes))
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
    # Mind the flatten() !!!
    ang1 = KX[0, :, :].flatten().T[:, np.newaxis] @ xm[0, :, :].flatten()[:, np.newaxis].T * param.omega
    ang2 = KX[1, :, :].flatten().T[:, np.newaxis] @ xm[1, :, :].flatten()[:, np.newaxis].T * param.omega
    phim = np.cos(ang1) * np.cos(ang2) * 2 ** (param.nbVarX)
    # Some weird +1, -1 due to 0 index !!!
    xx, yy = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
    phim = phim * np.tile(HK, (param.nbRes**param.nbVarX, 1)).T

    # Desired spatial distribution
    g = w_hat.T @ phim
    return g


class ErgodicControlHEDAC2D:

    def __init__(self):
        # Parameters
        # ===============================
        self.nbDataPoints = 500
        self.min_kernel_val = 1e-8  # upper bound on the minimum value of the kernel
        self.diffusion = 1  # increases global behavior
        self.source_strength = 1  # increases local behavior
        self.obstacle_strength = 0  # increases local behavior
        self.agent_radius = 10  # changes the effect of the agent on the coverage
        self.max_dx = 1  # maximum velocity of the agent
        self.max_ddx = 0.1  # maximum acceleration of the agent
        self.cooling_radius = 1  # changes the effect of the agent on local cooling (collision avoidance)
        self.nbAgents = 1
        self.local_cooling = 0  # for multi agent collision avoidance
        self.dx = 1

        self.nbVarX = 2  # dimension of the space
        self.nbResX = 100  # number of grid cells in x direction
        self.nbResY = 100  # number of grid cells in y direction

        self.nbGaussian = 2

        self.nbFct = 10  # Number of basis functions along x and y
        # Domain limit for each dimension (considered to be 1
        # for each dimension in this implementation)
        self.xlim = [0, 1]
        self.L = (self.xlim[1] - self.xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
        self.omega = 2 * np.pi / self.L

        self.nbRes = self.nbResX  # resolution of discretization

        self.alpha = np.array([1, 1]) * self.diffusion

        self.G = np.zeros((self.nbResX, self.nbResY))

        # Note this part is needed to have exact same target distribution as in ergodic_control_SMC.py
        # self.Mu, self.Sigma, self.Alpha = get_fixed_GMM(param)
        self.Mu, self.Sigma, self.Alpha = get_GMM(self)
        g = discrete_gmm(self)
        self.G = np.reshape(g, [self.nbResX, self.nbResY])
        self.G = np.abs(self.G)  # there is no negative heat

        self._init_agents()
        self._init_head_equation()

    def _init_agents(self):
        # Initialize agents
        # ===============================
        self.agents = []
        for i in range(self.nbAgents):
            # initial position of the agent
            # x0 = np.random.uniform(0, self.nbResX, 2)
            x0 = np.array([10, 30])  # if single agent same ic as SMC example
            agent = SecondOrderAgent(x=x0, nbDataPoints=self.nbDataPoints, max_dx=self.max_dx, max_ddx=self.max_ddx)
            # agent = FirstOrderAgent(x=x, dim_t=cfg.timesteps)
            rgb = np.random.uniform(0, 1, 3)
            agent.color = np.concatenate((rgb, [1.0]))  # append alpha value
            self.agents.append(agent)

    def _init_head_equation(self):
        # Initialize heat equation related fields
        # ===============================
        # precompute everything we can before entering the loop
        self.coverage_arr = np.zeros((self.nbResX, self.nbResY, self.nbDataPoints))
        self.heat_arr = np.zeros((self.nbResX, self.nbResY, self.nbDataPoints))
        self.local_arr = np.zeros((self.nbResX, self.nbResY, self.nbDataPoints))
        self.goal_arr = np.zeros((self.nbResX, self.nbResY, self.nbDataPoints))

        self.height, self.width = self.G.shape

        self.area = self.dx * self.width * self.dx * self.height

        self.goal_density = normalize_mat(self.G)

        self.coverage_density = np.zeros((self.height, self.width))
        self.heat = np.array(self.goal_density)

        max_diffusion = np.max(self.alpha)
        self.dt = min(
            1.0, (self.dx * self.dx) / (4.0 * max_diffusion)
        )  # for the stability of implicit integration of Heat Equation
        self.coverage_block = agent_block(self.min_kernel_val, self.agent_radius)
        self.cooling_block = agent_block(self.min_kernel_val, self.cooling_radius)
        self.kernel_size = self.coverage_block.shape[0]

    def run_from_init_pos(self, init_pos):
        # HEDAC Loop
        # ===============================
        # do absolute minimum inside the loop for speed
        trajectories = dict()
        for agent in self.agents:
            agent.x = init_pos
            trajectories[agent] = []
        for t in range(self.nbDataPoints):
            self.ergodic_step(t)

            for agent in self.agents:
                trajectories[agent].append(agent.x)

        return trajectories

    def ergodic_step(self, t):
        # cooling of all the agents for a single timestep
        # this is used for collision avoidance bw/ agents
        local_cooling = np.zeros((self.height, self.width))
        for agent in self.agents:
            # find agent pos on the grid as integer indices
            p = agent.x
            adjusted_position = p / self.dx
            col, row = adjusted_position.astype(int)

            # each agent has a kernel around it,
            # clamp the kernel by the grid boundaries
            row_indices, row_start_kernel, num_kernel_rows = clamp_kernel_1d(row, 0, self.height, self.kernel_size)
            col_indices, col_start_kernel, num_kernel_cols = clamp_kernel_1d(col, 0, self.width, self.kernel_size)

            # add the kernel to the coverage density
            # effect of the agent on the coverage density
            self.coverage_density[row_indices, col_indices] += self.coverage_block[
                row_start_kernel : row_start_kernel + num_kernel_rows,
                col_start_kernel : col_start_kernel + num_kernel_cols,
            ]

            # local cooling is used for collision avoidance between the agents
            # so it can be disabled for speed if not required
            # if self.local_cooling != 0:
            #     local_cooling[row_indices, col_indices] += cooling_block[
            #         row_start_kernel : row_start_kernel + num_kernel_rows,
            #         col_start_kernel : col_start_kernel + num_kernel_cols,
            #     ]
            # local_cooling = normalize_mat(local_cooling)
        coverage = normalize_mat(self.coverage_density)
        # this is the part we introduce exploration problem to the Heat Equation
        diff = self.goal_density - coverage
        sign = np.sign(diff)
        source = np.maximum(diff, 0) ** 2
        source = normalize_mat(source) * self.area
        current_heat = np.zeros((self.height, self.width))
        # 2-D heat equation (Partial Differential Equation)
        # In 2-D we perform this second-order central for x and y.
        # Note that, delta_x = delta_y = h since we have a uniform grid.
        # Accordingly we have -4.0 of the center element.
        # At boundary we have Neumann boundary conditions which assumes
        # that the derivative is zero at the boundary. This is equivalent
        # to having a zero flux boundary condition or perfect insulation.
        current_heat[1:-1, 1:-1] = self.dt * (
            (
                +self.alpha[0] * offset(self.heat, 1, 0)
                + self.alpha[0] * offset(self.heat, -1, 0)
                + self.alpha[1] * offset(self.heat, 0, 1)
                + self.alpha[1] * offset(self.heat, 0, -1)
                - 4.0 * offset(self.heat, 0, 0)
            )
            / (self.dx * self.dx)
            + self.source_strength * offset(source, 0, 0)
            - self.local_cooling * offset(local_cooling, 0, 0)
        ) + offset(self.heat, 0, 0)
        self.heat = current_heat.astype(np.float32)
        # Calculate the first derivatives mind the order x and y
        gradient_y, gradient_x = np.gradient(self.heat, 1, 1)
        for agent in self.agents:
            grad = self.calculate_gradient(
                agent,
                gradient_x,
                gradient_y,
            )
            local_heat = bilinear_interpolation(current_heat, agent.x)
            agent.update(grad)

        self.coverage_arr[..., min(t, self.nbDataPoints - 1)] = coverage
        self.heat_arr[..., min(t, self.nbDataPoints - 1)] = self.heat
        return coverage, self.heat

    def calculate_gradient(self, agent, gradient_x, gradient_y):
        """
        Calculate movement direction of the agent by considering the gradient
        of the temperature field near the agent
        """
        # find agent pos on the grid as integer indices
        adjusted_position = agent.x / self.dx
        # note x axis corresponds to col and y axis corresponds to row
        col, row = adjusted_position.astype(int)

        gradient = np.zeros(2)
        # if agent is inside the grid, interpolate the gradient for agent position
        if row > 0 and row < self.height - 1 and col > 0 and col < self.width - 1:
            gradient[0] = bilinear_interpolation(gradient_x, adjusted_position)
            gradient[1] = bilinear_interpolation(gradient_y, adjusted_position)

        # if kernel around the agent is outside the grid,
        # use the gradient to direct the agent inside the grid
        boundary_gradient = 2  # 0.1
        pad = self.kernel_size - 1
        if row <= pad:
            gradient[1] = boundary_gradient
        elif row >= self.height - 1 - pad:
            gradient[1] = -boundary_gradient

        if col <= pad:
            gradient[0] = boundary_gradient
        elif col >= self.width - pad:
            gradient[0] = -boundary_gradient

        return gradient

    def plot(self):
        # Plot
        # ===============================
        fig, ax = plt.subplots(1, 3, figsize=(16, 8))

        ax[0].set_title("Agent trajectory and desired GMM")
        # Required for plotting discretized GMM
        xlim_min = 0
        xlim_max = self.nbResX
        xm1d = np.linspace(xlim_min, xlim_max, self.nbResX)  # Spatial range
        xm = np.zeros((self.nbGaussian, self.nbResX, self.nbResY))
        xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
        X = np.squeeze(xm[0, :, :])
        Y = np.squeeze(xm[1, :, :])

        ax[0].contourf(X, Y, self.G, cmap="gray_r")  # plot discrete GMM
        # Plot agent trajectories
        for agent in self.agents:
            ax[0].plot(agent.x_arr[0, 0], agent.x_arr[0, 1], marker=".", color="black", markersize=10)
            ax[0].plot(
                agent.x_arr[:, 0],
                agent.x_arr[:, 1],
                color="black",
                linewidth=1,
            )
        ax[0].set_aspect("equal", "box")

        ax[1].set_title("Exploration goal (heat source), explored regions at time t")
        arr = self.goal_density - self.coverage_arr[..., -1]
        arr_pos = np.where(arr > 0, arr, 0)
        arr_neg = np.where(arr < 0, -arr, 0)
        ax[1].contourf(X, Y, arr_pos, cmap="gray_r")
        # Plot agent trajectories
        for agent in self.agents:
            ax[1].plot(
                agent.x_arr[:, 0], agent.x_arr[:, 1], linewidth=10, color="blue", label="agent footprint"
            )  # sensor footprint
            ax[1].plot(
                agent.x_arr[:, 0], agent.x_arr[:, 1], linestyle="--", color="black", label="agent path"
            )  # trajectory line
        ax[1].legend(loc="upper left")
        ax[1].set_aspect("equal", "box")

        ax[2].set_title("Gradient of the potential field")
        gradient_y, gradient_x = np.gradient(self.heat_arr[..., -1])
        ax[2].quiver(X, Y, gradient_x, gradient_y, scale=15, units="xy")  # Scales the length of the arrow inversely
        # ax[2].quiver(X, Y, gradient_x, gradient_y)

        # Plot agent trajectories
        for agent in self.agents:
            ax[2].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linestyle="--", color="black")  # trajectory line
            ax[2].plot(agent.x_arr[0, 0], agent.x_arr[0, 1], marker=".", color="black", markersize=10)
        ax[2].set_aspect("equal", "box")

        plt.show()
