#!/bin/bash
set -e

export PYTHONPATH=/media/tyshao/Jev-WM/experiments/robocasa_rulebased_4gpu_20261024/source:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

cd /media/tyshao/Jev-WM/experiments/robocasa_rulebased_4gpu_20261024

task=$1
if [ -z "$task" ]; then
  echo "Usage: bash run_4gpu.sh {train|sample|eval}"
  exit 1
fi

case $task in
  train)
    python -u -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_port=29600 \
      source/src/futureworlds/training.py \
      train.json 2>&1 | tee logs/train.log
    ;;
  sample)
    python source/src/futureworlds/training.py sampling.json 2>&1 | tee logs/sample.log
    ;;
  eval)
    python source/src/futureworlds/training.py evaluate.json 2>&1 | tee logs/eval.log
    ;;
  *)
    echo "Unknown task: $task"
    exit 1
    ;;
esac
