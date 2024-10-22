# Ergodic Control Project

## Project title: Trajectory Optimization for Ergodic Control Using Varying Imprint for Surface Finishing Applications


### Objective: 
Develop a software to generate effective trajectories for robotic surface finishing in the pybullet simulation using ergodic control strategies. We want to integrate iterative methods (iLQR) to the framework to find short trajectory that cover a given distribution as good as possible. Therefore we want to consider the imprint of the tool (tool area that is in contact with workpiece surface) instead of assuming a point contact and iteratively optimize the trajectory.

### Roadmap:

1. Familiarize with the iLQR method presented in the notebook [ilqr_ergodic_control.py](https://colab.research.google.com/github/MurpheyLab/ergodic-control-sandbox/blob/main/notebooks/ilqr_ergodic_control.ipynb#scrollTo=GpWFZQoi01c_) and the corresponding papers referenced there.

2. Integrate the method for a point mass in our framework e.g. use an closed-form ergodic control method (HEDAC/SMC) for generating a initial trajectory of a predefined fixed time horizon and use Trajectory Optimization method to improve the given trajectory iteratively in terms of ergodicity.

3. Visualize the output and execution of the robot in pybullet for different distributions (e.g. GMM, uniform distribution, target distribution from demonstration/image data).

4. Extend the approach by generalizing the point mass to a constant later variable footprint of the agent.

<img src="https://gitlab.lrz.de/i23-lectures/ws-2024-learning-robotic-skills-from-demonstration/learning-surface-finishing-from-demonstration/-/blob/main/src/ergodic_control/gifs/smc_ergodic_control.gif" alt="drawing" width="240"/>
<img src="https://gitlab.lrz.de/i23-lectures/ws-2024-learning-robotic-skills-from-demonstration/learning-surface-finishing-from-demonstration/-/blob/main/src/ergodic_control/gifs/ilqr_iters.gif" alt="drawing" width="250"/>
