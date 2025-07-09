#!/bin/sh
username="$USER"
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh
WORKDIR=/netscratch/$username/code/solution-yuxuan

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --job-name=code-prompt-sst2 \
  --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
  --container-workdir=$WORKDIR \
  --container-image=$IMAGE \
  --cpus-per-task=6 \
  --gpus-per-task=1 \
  --mem-per-cpu=4G \
  --ntasks=1 \
  --nodes=1 \
  $*
