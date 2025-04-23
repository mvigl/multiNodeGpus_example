#!/bin/bash

JOBTIME="00:10:00" # Wall clock limit (max. is 24 hours)
LOG_FILE="job_script.log"
ERR_FILE="job_script_error.log"

# Submit all jobs
lr=0.0001
bs=128
ep=100
gpus=4
out=models
project_name="multiGPUexample"
api_key="your comet_ml api key"
ws="your comet_ml workspace"

nohup sbatch --job-name="$mess" --time="${JOBTIME}" /raven/u/mvigl/multiNodeGpus_example/slurm/single_job.sbatch "$lr" "$bs" "$ep" "$bs" "$gpus" "$out" "$project_name" "$api_key" "$ws" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

# can add more submissions here
# lr=0.001
# nohup sbatch ......
# disown -h

exit 0
