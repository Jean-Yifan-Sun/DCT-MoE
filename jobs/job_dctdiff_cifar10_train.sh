#!/bin/bash
#SBATCH --account=qingjiem-heart-tte
#SBATCH --qos=bham
#SBATCH --time=48:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:2
#SBATCH --gpus-per-task 2
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_80
#SBATCH --mem=256G  # 请求内存
set -e
module purge
module load baskerville
module load bask-apps/live/live
# module load CUDA/11.3.1

# 运行 Python 命令
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda init
conda activate dctdiff
conda info --envs
cd /bask/projects/q/qingjiem-heart-tte/yifansun/project/DCTdiff
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1
# accelerate test
nohup accelerate launch --multi_gpu --mixed_precision fp16 train.py --config=configs/cifar10_uvit_mid_2by2.py --workdir output/cifar10_uvit_mid_2by2