#!/bin/bash -l
# Standard output and error:
#SBATCH -o /viper/u/mvigl/multiNodeGpus_example/slurm/job-management/job_%j.out
#SBATCH -e /viper/u/mvigl/multiNodeGpus_example/slurm/job-management/job_%j.err
# Initial working directory:
#SBATCH -D /viper/u/mvigl/multiNodeGpus_example
#


# 2 GPUs on a full node ---
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=120000
#
# uncomment if you want job updates by email, change user
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=<user>@mpcdf.mpg.de

# Load compiler and MPI modules
module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0
module load python-waterboa/2024.06

pip install --no-build-isolation -e .
pip install -r requirements_AMD.txt 

export OMP_NUM_THREADS=24
export COMET_OFFLINE_DIRECTORY='comet_offline_log'

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
backend="$9"
# Start run
srun /viper/u/mvigl/multiNodeGpus_example/slurm/job_viper.sh "$lr" "$bs" "$ep" "$gpus" "$out" "$project_name" "$api_key" "$ws" "$backend"
