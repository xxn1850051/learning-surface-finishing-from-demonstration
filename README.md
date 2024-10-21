# Masterpraktikum: Learning Robotic Skills from Demonstration » Lernen robotischer Skills aus Demonstrationen
This code is only to be used for teaching purposes, please do not distribute it. Standard copyright laws apply.

## Prepare the environment
We assume a standard Ubuntu installation. Simply execute `./create_venv.sh` to create a virtual environment with all Python 3.11 packages you need. Then, you can execute `source .pybullet/bin/activate` in your shell to use the environment to execute the code.

### Run the dynamical system code
Execute `python3 -m src.dynamical_systems.kmp_dynamical_system` to execute the code. This will first show a plot of the recorded data, once you close that, pybullet will open and the code will send velocity commands to the simulated robot.

### Run the ergodic control code
See the src/ergodic_control subdirectory.

### Run the interactive learning code
Execute `python3 -m src.interactive_learning.kmp_interactive_learning`