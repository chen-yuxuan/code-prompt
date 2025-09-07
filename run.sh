#!/bin/sh
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_25.01-py3.sqsh

srun -K \
  --container-mounts="`pwd`":"`pwd`" \
  --job-name="code-prompt" \
  --export=ALL,HF_HUB_CACHE=/netscratch/$USER/code/code-prompt/data, \
  --container-workdir="`pwd`" \
  --container-image=$IMAGE \
  --container-mounts=/netscratch/$USER/code/code-prompt:/netscratch/$USER/code/code-prompt,/ds:/ds,`pwd`:`pwd` \
  --cpus-per-task=6 \
  --gpus-per-task=1 \
  --mem-per-cpu=8G \
  --ntasks=1 \
  --nodes=1 \
  $*
