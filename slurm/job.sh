#!/bin/bash

lr="$1"
bs="$2"
ep="$3"
gpus="$4"
out="$5"
project_name="$6"
api_key="$7"
ws="$8"

python -m torch.distributed.run --nproc_per_node="$gpus" \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /raven/u/mvigl/multiNodeGpus_example/training.py \
    --lr "$lr" \
    --bs "$bs" \
    --ep "$ep" \
    --out "$out" \
    --project_name "$project_name" \
    --api_key "$api_key" \
    --ws "$ws" \