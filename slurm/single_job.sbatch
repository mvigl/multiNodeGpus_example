#!/bin/bash -l
# Standard output and error:
#SBATCH -o /raven/u/mvigl/multiNodeGpus_example/slurm/job-management/job_%j.out
#SBATCH -e /raven/u/mvigl/multiNodeGpus_example/slurm/job-management/job_%j.err
# Initial working directory:
#SBATCH -D /raven/u/mvigl/multiNodeGpus_example
#


# 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
#
# uncomment if you want job updates by email, change user
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=<user>@mpcdf.mpg.de

# Load compiler and MPI modules
module purge
# python 3.10.
module load anaconda/3/2023.03 
module load cuda/12.6

pip install -e .
pip install -r requirements.txt 

export OMP_NUM_THREADS=18

# Get list of nodes and master address
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
export MASTER_ADDR=${nodes_array[0]}
export MASTER_PORT=29500  # any free port


lr="$1"
bs="$2"
ep="$3"
gpus="$4"
out="$5"
project_name="$6"
api_key="$7"
ws="$8"
# Start run
srun /raven/u/mvigl/multiNodeGpus_example/slurm/job.sh "$lr" "$bs" "$ep" "$gpus" "$out" "$project_name" "$api_key" "$ws"
