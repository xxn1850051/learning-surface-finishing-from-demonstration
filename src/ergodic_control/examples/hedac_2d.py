import numpy as np

from src.ergodic_control.examples.hedac_2d_online import state_to_3d
from src.libs.ergodic_control_HEDAC_2D import ErgodicControlHEDAC2D
from src.libs.sim import Simulator

if __name__ == "__main__":
    erg_ctrl = ErgodicControlHEDAC2D()
    trajs = erg_ctrl.run_from_init_pos(np.array([10, 30]))
    erg_ctrl.plot()

    input("Simulator (press enter to continue)")
    sim = Simulator()
    sim.setup_scenario(local=True, tool="grinder")
    agent = erg_ctrl.agents[0]

    def pos_controller(t):
        traj = trajs[agent]
        i = min(int(t / sim.time_step), len(traj) - 1)
        traj_pt = state_to_3d(traj[i])
        print(f"time: {t} pos: {traj_pt.T}")
        return traj_pt

    sim.add_traj_pos_ctl(pos_controller)
    input("(press enter to quit)")
