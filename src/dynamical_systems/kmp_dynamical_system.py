import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from src.libs.data import read_frames
from src.libs.gmm import GaussianMixtureModel
from src.libs.kmp import Kmp
from src.libs.plot import mplot3d_add_ellipsoid
from src.libs.sim import Simulator

# first read the data and calculate velocities
dataset = []
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
for file in os.listdir(datadir):
    if file.endswith(".log"):
        data = read_frames(os.path.join(datadir, file))
        time = data[:, 0]
        position = data[:, 1:4]
        velocity = (position[1:, :] - position[:-1, :]) / (time[1:] - time[:-1]).reshape(-1, 1)
        joined_data = np.hstack((position[:-1, :], velocity))
        dataset.append(joined_data)
dataset = np.vstack(dataset)

# create the gmm
gmm = GaussianMixtureModel(n_components=5)
gmm.fit(dataset)

# marginalize gmm to position part for covariance extraction
marg_gmm = gmm.gaussian_margin([0, 1, 2])

# extract kmp data
pos = dataset[::10, :3]
cov_pos = np.array([marg_gmm.approximate_single_gaussian(weight)[1] for weight in marg_gmm.predict_proba(pos)])
vel, cov_vel, _ = gmm.gaussian_mixture_regression(pos, [0, 1, 2], [3, 4, 5])

# fit the kmp
kmp = Kmp(gmm_n_components=gmm.n_components, N=vel.shape[0])
kmp.x_in = pos
kmp.nb_dim_in = 3
kmp.mu = vel
kmp.mu_block = vel.reshape(-1, 1)
kmp.nb_dim_out = 3
kmp.l = kmp.l = np.ones(kmp.nb_dim_in)
kmp.sigma_block = scipy.linalg.block_diag(*[cov_pos[i] for i in range(cov_pos.shape[0])])
kmp.update_K()

# query the kmp
vel_kmp = kmp.predict(pos)[0]

# plot data
ax = plt.figure().add_subplot(projection="3d")
ax.quiver(
    dataset[::10, 0],
    dataset[::10, 1],
    dataset[::10, 2],
    dataset[::10, 3],
    dataset[::10, 4],
    dataset[::10, 5],
    length=0.1,
    normalize=True,
    color="b",
)
ax.quiver(
    dataset[::10, 0],
    dataset[::10, 1],
    dataset[::10, 2],
    vel_kmp[:, 0],
    vel_kmp[:, 1],
    vel_kmp[:, 2],
    length=0.1,
    normalize=True,
    color="r",
)

for i in range(marg_gmm.n_components):
    mplot3d_add_ellipsoid(ax, marg_gmm.means_[i, :], marg_gmm.covariances_[i])
plt.show()

# start simulation
sim = Simulator()
sim.setup_scenario([-0.367, 0.993, -0.392, -1.164, 0.506, 1.070, -1.147])

# -- start velocity control
input("Press Enter to start the velocity controller")
while True:
    pose = sim.get_cartesian_pose()
    cmd = np.zeros(6)
    cmd[:3] = kmp.predict(pose[:3])[0]
    # TODO fuse with base policy / TP policy
    print(f"Applying velocity command {cmd}")
    sim.cartesian_velocity_control(cmd)
