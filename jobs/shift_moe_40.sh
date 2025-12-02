#!/bin/bash
#SBATCH --account=qingjiem-heart-tte
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs_shift/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l6r16.py --workdir output_shift/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l6r16_minmax

accelerate launch --multi_gpu --num_processes=2 --mixed_precision=no train_greyscale_MoE.py \
    --config=configs_shift/acdc_wholeheart_uncond_uvit_-1MoE_greyscale_mid_4by4.py \
    --workdir output_shift/acdc_wholeheart_uncond_-1MoE_greyscale_uvit_mid_4by4_Y_bound

# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=no train_greyscale_MoE.py \
#     --config=configs_shift/echonet_dynamic_uncond_uvit_-1MoE_greyscale_mid_4by4.py \
#     --workdir output_shift/echonet_dynamic_uncond_-1MoE_greyscale_uvit_mid_4by4_Y_bound