#!/bin/bash
#SBATCH --account=chenhp-data-gen
#SBATCH --qos=bham
#SBATCH --time=128:00:00
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
cd /bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1

# accelerate launch --multi_gpu --mixed_precision=no train_ldm.py --config=configs_ldm/acdc_wholeheart_uncond_uvit_mid.py --workdir=output_ldm/acdc_wholeheart_uncond_uvit_mid

# accelerate launch --multi_gpu --mixed_precision=no train_ldm.py --config=configs_ldm/acdc_wholeheart_uncond_uvit_mid_p2.py --workdir=output_ldm/acdc_wholeheart_uncond_uvit_mid_p2

accelerate launch --multi_gpu --mixed_precision=no train_ldm.py --config=configs_ldm/echonet_uncond_uvit_mid_p2.py --workdir=output_ldm/echonet_uncond_uvit_mid_p2