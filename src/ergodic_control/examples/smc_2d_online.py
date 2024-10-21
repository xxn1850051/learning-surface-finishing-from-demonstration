from copy import copy

import numpy as np

from src.ergodic_control.examples.hedac_2d_online import state_to_3d
from src.libs.ergodic_control_SMC_2D import ErgodicControlSMC2D
from src.libs.sim import Simulator

if __name__ == "__main__":
    erg_ctrl = ErgodicControlSMC2D()
    sim = Simulator()
    sim.setup_scenario(local=True, tool="grinder")
    x = np.array([0.1, 0.3])

    def pos_controller(t):
        global x
        i = int(t / sim.time_step)
        w, x = erg_ctrl.ergodic_step(i, x)
        print(x)
        traj_pt = state_to_3d(x, scale=0.6)
        print(f"time: {t} pos: {traj_pt.T}")
        return traj_pt

    sim.add_traj_pos_ctl(pos_controller, time_horizon=np.inf)
    input("(press enter to quit)")
