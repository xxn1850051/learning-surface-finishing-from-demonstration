from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def mplot3d_add_ellipsoid(
    ax: Axes3D, center: np.ndarray, sigma: np.ndarray, nb_arcs: int = 100, color: Optional[str] = None
) -> None:
    """
    Plot an ellipsoid in a 3D plot

    :param ax: The axes object to plot onto.
    :param center: (3,) center of the ellipsoid.
    :param sigma: (3,3) covariance matrix of the ellipsoid.
    :paam nb_arcs: Number of arc segments used for each DoF (u, v) to plot the ellipsoid.
    :param color: The color to plot the ellipse in, defaults to black.
    """

    if color is None:
        color = "k"

    assert center.shape[0] == 3 and len(center.shape) == 1
    assert sigma.shape[0] == 3 and sigma.shape[1] == 3 and len(sigma.shape) == 2

    # find the rotation matrix and radii of the axes
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    # make right hand if needed
    if np.linalg.det(eigenvectors) < 0:
        # https://github.com/ros-visualization/rviz/blob/melodic-devel/src/rviz/default_plugin/covariance_visual.cpp#L53
        eigenvectors = eigenvectors[:, [1, 0, 2]]
        eigenvalues = eigenvalues[[1, 0, 2]]

    radii = np.sqrt(eigenvalues)

    # Build unit sphere and transform into coordinates
    u = np.linspace(0.0, 2.0 * np.pi, nb_arcs)
    v = np.linspace(0.0, np.pi, nb_arcs)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eigenvectors.T) + center

    # plot
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=color, alpha=0.2)


def kmp_2dplot(kmp, mu_out, data_base, via_points=None):
    # KMP 2d plot with covariance
    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle("KMP reproduction - mean", fontsize=16)
    for i in range(3):
        plt.subplot(3, 1, 1 + i)
        plt.scatter(data_base[:, 0], data_base[:, i + 1], c="b", label="demonstrations")
        plt.scatter(kmp.x_in[: kmp.mu.shape[0], 0], kmp.mu[:, i], c="g", label="ref traj (GMR)")
        plt.scatter(kmp.x_in[: mu_out.shape[0], 0], mu_out[:, i], c="r", label="kmp traj")

        variances = kmp.diag_blocks(kmp.cov())
        std = np.sqrt(variances) * 5  # this factor is just for visualization purposes

        upper_line = mu_out[:, i] + std[:, i, i]
        under_line = mu_out[:, i] - std[:, i, i]
        plt.fill_between(
            x=kmp.x_in[: kmp.mu.shape[0], 0],
            y1=under_line,
            y2=upper_line,
            alpha=0.3,
            color="#080808",
        )

        if via_points is not None:
            plt.scatter(via_points[:, 0], via_points[:, i + 1], c="y", label="Via points")
    plt.legend()
    plt.show()
