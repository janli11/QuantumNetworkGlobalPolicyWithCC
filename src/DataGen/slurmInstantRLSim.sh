#!/bin/bash -l
#SBATCH --array=0-359
#SBATCH -o ./tjob.%A_%a.out
#SBATCH -e ./tjob.%A_%a.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J mdp_comparison # job dependent
# Queue (Partition):
#SBATCH --partition=cpu-short
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=janli@mail.lorentz.leidenuniv.nl # useslurr dependent
#
# Wall clock limit
#SBATCH --time=3:59:00

# asking for more memory
#SBATCH --mem=4G

#Load some modules
module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.11-gfbf-2023b
source /home/lijt/quantumNetworkInhom/rl_env/bin/activate

# adding paths for my modules
export PYTHONPATH="$PYTHONPATH:~/quantum_network/code_for_paper/src"
export PYTHONPATH="$PYTHONPATH:~/quantum_network/code_for_paper/src/environments"
export PYTHONPATH="$PYTHONPATH:~/quantum_network/code_for_paper/src/TrainSim"

# Run the program:
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Total number of tasks: " $SLURM_ARRAY_TASK_COUNT
N_idx=$SLURM_ARRAY_TASK_COUNT

python DataGen/InstantRLTrainSimHpc.py -idx $SLURM_ARRAY_TASK_ID -N_idx $N_idx -train 1 -train_more 0 -train_steps "1e7" -sim 1 -sim_eps "1e4" -train_new_model 1 -training_version_start "0" -training_version_stop "20"
