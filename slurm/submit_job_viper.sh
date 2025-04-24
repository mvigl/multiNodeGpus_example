#!/bin/bash

JOBTIME="00:05:00" # Wall clock limit (max. is 24 hours)
LOG_FILE="job_script.log"
ERR_FILE="job_script_error.log"

# Submit all jobs
lr=0.0001
bs=128
ep=3
gpus=2
out=models
project_name="multiGPUexample"
api_key="your comet_ml api key"
ws="your comet_ml workspace"
backend='gloo'

nohup sbatch --job-name="test" --time="${JOBTIME}" /viper/u/mvigl/multiNodeGpus_example/slurm/single_job_viper.sbatch "$lr" "$bs" "$ep" "$gpus" "$out" "$project_name" "$api_key" "$ws" "$backend" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

# can add more submissions here
# lr=0.001
# nohup sbatch ......
# disown -h

exit 0
