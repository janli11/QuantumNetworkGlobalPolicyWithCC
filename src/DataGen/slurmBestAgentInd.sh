#!/bin/bash -l
#SBATCH -o ./tjob.%A.out
#SBATCH -e ./tjob.%A.err
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
#SBATCH --time=2:59:00

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

python DataGen/bestAgentFinderIndHpc.py 
