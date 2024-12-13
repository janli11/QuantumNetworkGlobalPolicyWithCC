Reinforcement learning for entanglement distribution policies with CC effects. 
======================================

This repository contains the code to simulate the quantum network, code for the reinforcement learning environment and the training, as well as code to simulate the trained policy and the plotting of the figures. 


## Installation
The code is written for Python 3 and tested to work with Python  3.11.3.

To make sure that all requirements of the package are fulfilled, the easiest way to use the code is to create a virtual environment and install the necessary packages independent of the python packages of the operating system

1. Creation of a virtual environment  
You can create the environment in a folder of your choice. 
For the rest of the tutorial, we assume it to be in `~/.pyenv/`
```
cd ~/.pyenv
python -m venv RLQNenv
```
Assuming you are using bash or zsh, you can activate the environment with `source ~/.pyenv/RLQNenv/bin/activate`.
Upon activation, you will notice that your prompt changes.
As long as it is prefixed with `(RLQNenv)` the virtual environment is active.
The virtual environment can be deactivated with `deactivate`.

2. Cloning the code  
You can obtain the code by cloning the repo with
```
git clone https://github.com/janli11/QuantumNetworkGlobalPolicyWithCC.git
```

Note that you have to be a member of the project to clone it.
Cloning via SSH works only if you have added a (public) SSH key to the repository.

3. Preparation of the environment  
For the next step, please navigate into the repo that you just downloaded and activate the empty environment that we created in step 1 with `source ~/.pyenv/RLQNenv/bin/activate`.
In order to install all required packages for the simulation, execute
```
pip install -r requirements.txt
```

4. Modification of the PYTHONPATH  
Since we are working with a package, we have to add it to the PYTHONPATH in order to make Python aware of its existence.
We add the repository to the PYTHONPATH with the following command
```export PYTHONPATH=$PYTHONPATH:<path_to_repo>```
where `<path_to_repo>` is the location of the repository on your computer.

If you want to make the change persistent, i.e. it remains after closing the terminal, you can consider adding it to your `~/.bashrc` (for bash) or `~/.zshrc` (for zsh).

## Structure for the Code

The repository is split into three main parts: the source code (src), data generations scripts (DataGen) and the plotting scripts (plotting).

### Source code

#### Quantum Network Simulator and Markov Decision process environment 

The 'qnetcc' package is divided into two parts. The first part, 'environments', contains all of the necessary code to construct our gynamsium (https://gymnasium.farama.org/) enviroment. The second part 'train_and_sim' contains all of the necessary code to train our agent on the gymnasium environment and to simulate the trained agent as well as some fixed policies. For the training of the agent, we are using 'PPO' from Stable-Baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

The `environments` package is divided into several modules as follows:
- ActionSpace.py
- History.py
- MDPEnv.py
- MDPRootClass.py
- ObsSpace.py
- QNSimulator (subdirectory)

The MDPEnv.py module is the module where the MDP environemnt is defined on which the agent is trained. It is a child class of the MDPRootClass.py, ActionSpace.py and ObsSpace.py modules. 
The QNSimulator contains the modules: 
- node.py
- qubit.py
- QuantumNetwork.py
This directory provides a simulator for the quantum network which is used in the MDP environment. 

The `TrainSim` package is divided into several modules as follows:
- MDPTrainSim.py
- MyCallbacks.py
- RandPolSim.py
- SwapAsapPredObsSim.py
- SwapAsapInstantSim.py
- SwapAsapVanillaSim.py

#### Training and simulation 

The module 'MDPTrainSim.py' contains functions for training and simulating the Reinforcement learning agent on the enviroment defined in 'MDPEnv.py'.

The module 'MyCallbacks.py' contains stable-baselines3 callbacks to monitor the training and save the model whenever a new best has been found. 

The module 'SwapAsapInstantSim.py' contains functions for simulating the instantaneous swap-asap policy.

The module 'SwapAsapPredObsSim.py' contains functions for simulating the predictive swap-asap policy. 

The module 'SwapAsapVanillaSim.py' contains functions for simulating the vanilla swap-asap policy. 

### Data Generation

The modules for training various RL models at different network parameters and simulating the RL as well as various swap-asap policies can be found in the 'DataGen' folder. 

The module 'PolicyComparison.py' is primairly intended for locally training the RL agent and simulating it together with the swap-asap policies to compare them with each other. 

To train the files on a cluster, please modify the slurm files to your specific needs. The corresponding slurm files will run the 'SwapAsapSimHpc.py' and the 'RLTrainSimHpc.py' modules.  

In the shell scripts 'slumRLTrain.sh' and 'slumRLSim.sh', the last line 

'''
python DataGen/RLTrainSimHpc.py -idx $SLURM_ARRAY_TASK_ID -N_idx $N_idx -train ... -train_more ... -train_steps ... -sim ... -sim_eps "..." -train_new_model ... -training_version_start ... -training_version_stop ...
'''

Is used to launch the training and/or simulation of the RL agent. It takes the following arguments at 
-train: 0 or 1; 0 to not train the model, 1 to train the model.  
-train_more: 0 or 1; 0 to not train if a trained model already exists, 1 to continue training the model.  
-train_steps: str; string specifiying the number of steps to train, e.g. "1e5". Will be converted to a float in the RLTrainSimHPc.py module.
-sim: 0 or 1; 0 to not simulate the policy, 1 to simulate the policy and save the delivery times. 
-sim_eps: str; string specifiying the number of steps to simulate the RL policy, e.g. "1e5". Will be converted to a float in the RLTrainSimHPc.py module.
-train_new_model: 0 or 1; Whether to train a new version of the model if there currently already exists one
-train_version_start: int(positive); Trains a model at the actual labelled by a specific version indices. This is the start index. 
-train_version_stop: int(positive); Trains a model at the actual labelled by a specific version indices. This is the end index. 


### Plotting

The plots in the paper are made using the scripts in the plotting folder. The heatmaps of Figure 6 is made with Figure6.py. Figure 7 of the paper is made with Figure7.py.


## Data used in paper [https://arxiv.org/abs/2412.06938]

The Trained models and simulation data used to create the figures are stored in a separate gitlab repository: https://gitlab.strw.leidenuniv.nl/janli/quantumnetworkglobalpolicywithccdata.git

To recreate Figure 7 and the heatmaps of Figure 6, please first download the data from the github repository. Then move it into a folder called data on the same level as the code from https://github.com/janli11/QuantumNetworkGlobalPolicyWithCC.git. Running the scripts Figure6.py and Figure7.py should reproduce the desired figures. 







