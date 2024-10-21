import os
import time
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data as pd

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class Simulator:

    def __init__(self):
        self.robot = None
        self.robot_dofs = None
        self.robot_ee_index = None  # robot end effector index
        self.ll = None
        self.ul = None
        self.jr = None
        self.rp = None

        self.flags = None
        self.table = None
        self.floor = None

        self.time_step = None

    # Configurations related to the type of robot we're using
    def init_robot_configuration(self, jp):
        index = 0
        for j in range(p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            # From Quick Start Guide: "Also note that calculateInverseDynamics ignores the
            # joint/link damping, while forward dynamics (in stepSimulation)
            # includes those damping terms. So
            # if you want to compare the inverse dynamics and forward dynamics, make sure to set those
            # damping terms to zero using changeDynamics with jointDamping and link damping through
            # linearDamping and angularDamping"
            #
            # I wonder how this affects the friction when moving in gravity comp.

            info = p.getJointInfo(self.robot, j)

            # jointName = info[1]
            jointType = info[2]
            # print(jointType)

            # we care about the joints that are not of type "fixed"
            if jointType == p.JOINT_PRISMATIC:
                p.resetJointState(self.robot, j, jp[index])
                index = index + 1
            if jointType == p.JOINT_REVOLUTE:
                p.resetJointState(self.robot, j, jp[index])
                index = index + 1

        self.robot_dofs = index

    # Reset to initial configuration
    def reset_robot(self, pybullet_sim, robot_id, reset_config):
        t = 0
        dt = p.getPhysicsEngineParameters()["fixedTimeStep"]  # stick to this to avoid carrying dt definitions around
        while t < 2.0:
            NumDofs = 7
            pybullet_sim.setJointMotorControlArray(
                robot_id, range(0, NumDofs), p.POSITION_CONTROL, targetPositions=reset_config[:NumDofs], forces=[50] * NumDofs
            )
            pybullet_sim.stepSimulation()
            time.sleep(dt)
            t += dt
        self.robot_dofs = 7
        torques = np.array(self.robot_dofs * [100]) * 0

    def setup_scenario(self, initial_joint_positions: Optional[list] = None, tool=None, local=True):
        # -- connect to physics server
        if p.isConnected():
            p.disconnect()
        p.connect(p.GUI)
        # -- simulator config
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # toggle to see world frame, grid, etc
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2,
            cameraYaw=126.8,
            cameraPitch=-34.6,
            cameraTargetPosition=(0.08100888878107071, -0.13343113660812378, -0.7280009984970093),
        )
        if local:
            p.setAdditionalSearchPath(datadir + "/urdf")
        else:
            p.setAdditionalSearchPath(pd.getDataPath())
        p.setRealTimeSimulation(True)  # removes the need to use p.stepSimulation()
        self.time_step = 1.0 / 60.0
        p.setTimeStep(self.time_step)  # there is a default value if I remove this
        p.setGravity(0, 0, -9.8)
        offset = [0, 0, 0]
        # -- initialize experimental setup
        self.robot_ee_index = 6
        self.robot_dofs = 7
        self.ll = [-7] * self.robot_dofs  # upper limits for null space (todo: set them to proper range)
        self.ul = [7] * self.robot_dofs  # joint ranges for null space (todo: set them to proper range)
        self.jr = [7] * self.robot_dofs  # rest poses for null space
        if initial_joint_positions is None:
            initial_joint_positions = [2.0, 0.458, 0.31, -2.24, -0.30, 0.4, 1.32]
        self.rp = initial_joint_positions
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        # -- add objects and robot
        self.floor = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.65], useFixedBase=False)
        self.table = p.loadURDF(
            "table/table.urdf",
            [0.5, 0.35, -0.625],
            p.getQuaternionFromEuler([0.0, 0.0, 0]),
            useFixedBase=False,
            flags=self.flags,
        )
        # input("Press Enter to add robot")
        orn1 = p.getQuaternionFromEuler([0.0, 0.0, -np.pi / 2.0])  # base orientation wrt world frame
        if tool is None:
            self.robot = p.loadURDF(
                "kuka_iiwa/model.urdf", np.array([0.0, 0.0, 0.0]), orn1, useFixedBase=True, flags=self.flags
            )
        elif tool == "grinder":
            self.robot = p.loadURDF(
                "kuka_iiwa/model_with_grinder.urdf", np.array([0.0, 0.0, 0.0]), orn1, useFixedBase=True, flags=self.flags
            )
        else:
            raise ValueError(f"tool {tool} unknown")
        self.init_robot_configuration(initial_joint_positions)

    def add_example_objects(self):
        legos = []
        legos.append(p.loadURDF("lego/lego.urdf", np.array([0.3, 0.2, 0.1]), flags=self.flags))
        legos.append(p.loadURDF("lego/lego.urdf", np.array([0.3, 0.15, 0.1]), flags=self.flags))
        legos.append(p.loadURDF("lego/lego.urdf", np.array([0.3, 0.3, 0.1]), flags=self.flags))
        sphereId = p.loadURDF("sphere_small.urdf", np.array([0.1, 0.3, 0.2]), flags=self.flags)
        p.loadURDF("sphere_small.urdf", np.array([0.2, 0.2, 0.2]), flags=self.flags)
        p.loadURDF("sphere_small.urdf", np.array([0.15, 0.15, 0.2]), flags=self.flags)

    def add_example_traj(self, calc_point=None):
        orn_h = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])  # fixed e.e. orientation
        print(orn_h)

    def move_robot(self, x_h):
        # orn_h = np.array([0, 0, -0.7071, 0.7071])
        # orn_h = np.array([0.5, 0.5, 0.5, 0.5])
        orn_h = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])  # fixed e.e. orientation
        jointPoses = p.calculateInverseKinematics(
            self.robot, self.robot_ee_index, x_h, orn_h, self.ll, self.ul, self.jr, self.rp, maxNumIterations=5
        )
        # send position commands to joints
        for i in range(self.robot_dofs):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, jointPoses[i], force=5 * 240.0)

    # def close_grippers(self, p, iiwa1):
    #    p.setJointMotorControl2(iiwa1, 9, p.POSITION_CONTROL, 0.0, force=5 * 2)
    #    p.setJointMotorControl2(iiwa1, 10, p.POSITION_CONTROL, 0.0, force=5 * 2)

    def draw_motion_as_line(self, time_step, x_prev):
        x = p.getLinkState(self.robot, 7)  # Link 9 is the end-effector
        if time_step > 0.0:
            p.addUserDebugLine(x_prev, x[0], [255 / 256, 99 / 256, 71 / 256], lineWidth=3.0)
        x_prev = x[0]
        time_step += self.time_step
        time.sleep(self.time_step)
        return time_step, x_prev

    def add_traj_pos_ctl(self, pos_controller=None, time_horizon=20.0):
        if pos_controller is None:
            f = 0.1  # frequency of signal

            def pos_controller(t):
                x_h = np.zeros((3, 1))
                x_h[2] = 0.4  # Fixed height, draw on the plane
                x_h[0] = 0.2 * np.cos(f * 2 * np.pi * t) + 0.4
                x_h[1] = 0.2 * np.sin(f * 2 * np.pi * t) + 0.4
                return x_h

        t = 0
        x_prev = 0
        while t < time_horizon:
            x_h = pos_controller(t)
            self.move_robot(x_h=x_h)
            t, x_prev = self.draw_motion_as_line(time_step=t, x_prev=x_prev)

    def clear(self):
        p.removeAllUserDebugItems()

    def get_cartesian_pose(self):
        (
            linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
        ) = p.getLinkState(self.robot, 6, 0, 1)

        return np.hstack((linkWorldPosition, linkWorldOrientation))

    def cartesian_velocity_control(self, velocity_command):
        q = [p.getJointState(self.robot, motor_id)[0] for motor_id in range(7)]
        linearJacobian, angularJacobian = p.calculateJacobian(
            self.robot, 6, [0, 0, 0], q, np.zeros(7).tolist(), np.zeros(7).tolist()
        )
        jacobian = np.zeros((6, 7))
        jacobian[:3, :] = linearJacobian
        jacobian[3:, :] = angularJacobian

        q_dot = np.linalg.pinv(jacobian) @ velocity_command

        for joint, velocity in zip(range(7), q_dot):
            p.setJointMotorControl2(self.robot, joint, p.VELOCITY_CONTROL, targetVelocity=velocity, force=100)


if __name__ == "__main__":
    sim = Simulator()
    sim.setup_scenario()

    # -- add some objects
    input("Press Enter to add objects")
    sim.add_example_objects()

    # -- toy tracking of periodic signal
    input("Press Enter to move")
    sim.add_traj_pos_ctl()

    input("Press Enter to quit.")
