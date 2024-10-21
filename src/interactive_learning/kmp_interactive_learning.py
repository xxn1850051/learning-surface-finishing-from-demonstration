import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.libs.data import read_frames
from src.libs.kmp import Kmp
from src.libs.plot import kmp_2dplot
from src.libs.sim import Simulator


def get_data():
    # first read the data and calculate velocities
    dataset = []
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    for file in os.listdir(datadir):
        if file.endswith(".pickle"):
            with open(os.path.join(datadir, file), "rb") as input_file:
                dataset = pickle.load(input_file)
    dataset = np.vstack(dataset)
    dataset = dataset[:, :4]  # only keep time, x, y, z

    # normalize_time
    T_max = np.max(dataset[:, 0])  # demonstrations go from t=0 to t=T_max
    # Input between 0-1.0 (comment out below if not)
    dataset[:, 0] /= T_max
    return dataset


if __name__ == "__main__":

    dataset = get_data()

    nb_dim_in = 1
    nb_dim_out = 3

    kmp_kwargs = {
        "gmm_n_components": 12,
        "N": 500,
        "l": 0.1,
        "h": 1.0,
        "lambda1": 0.1,
        "lambda2": 1,
        "alpha": 1,
        "kernel_function": "matern2",
    }

    kmp = Kmp(**kmp_kwargs)

    # Initialize trajectory distribution
    t_lim = 1.0  # T_max
    x_in = np.linspace(0, t_lim, kmp.N)
    kmp.fit(dataset, [0], [1, 2, 3], x_in=x_in)

    # query the kmp
    pos_kmp = kmp.predict(x_in)[0]
    # plot data
    # kmp_2dplot(kmp=kmp, mu_out=pos_kmp, data_base=dataset)

    # add via-points
    via_point_times = np.array([0.5, 0.6])
    via_points = np.array([[0.53, 0.1, 0.2], [0.53, 0.1, 0.2]])
    kmp.add_viapoints(input_via=via_point_times, output_via=via_points, gamma=1e-12)

    # predict again with via-points
    pos_kmp = kmp.predict(x_in)[0]
    # plot data
    # kmp_2dplot(kmp=kmp, mu_out=pos_kmp, data_base=dataset, via_points=np.concatenate([via_point_times[:, None], via_points], axis=1))

    # start simulation
    # input("Simulator (press enter to continue)")
    sim = Simulator()
    sim.setup_scenario(local=True, tool="grinder")

    def pos_controller(t):
        i = int(t / sim.time_step) % pos_kmp.shape[0]
        return pos_kmp[i].reshape((3, 1))

    sim.add_traj_pos_ctl(pos_controller, time_horizon=np.inf)
