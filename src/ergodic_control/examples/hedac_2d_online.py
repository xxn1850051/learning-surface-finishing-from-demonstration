import numpy as np

from src.libs.ergodic_control_HEDAC_2D import ErgodicControlHEDAC2D
from src.libs.sim import Simulator


def state_to_3d(x, offset=np.array([0.05, 0.05]), scale=0.006):
    traj_pt = np.zeros((3, 1))
    traj_pt[:2, 0] = scale * x + offset
    traj_pt[2] = 0.3  # Fixed height, draw on the plane
    return traj_pt


if __name__ == "__main__":
    erg_ctrl = ErgodicControlHEDAC2D()
    sim = Simulator()
    sim.setup_scenario(local=True, tool="grinder")
    agent = erg_ctrl.agents[0]
    agent.x = np.array([10, 30])

    def pos_controller(t):
        i = int(t / sim.time_step)
        erg_ctrl.ergodic_step(i)
        pos = state_to_3d(agent.x)
        print(f"time: {t} pos: {pos.T}")
        return pos

    sim.add_traj_pos_ctl(pos_controller, time_horizon=np.inf)
    input("(press enter to quit)")
