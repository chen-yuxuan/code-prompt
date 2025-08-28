#!/bin/sh
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.12-py3.sqsh

srun -K \
  --container-mounts="`pwd`":"`pwd`" \
  --job-name="code-prompt" \
  --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
  --container-workdir="`pwd`" \
  --container-image=$IMAGE \
  --cpus-per-task=6 \
  --gpus-per-task=1 \
  --mem-per-cpu=20G \
  --ntasks=1 \
  --nodes=1 \
  $*
