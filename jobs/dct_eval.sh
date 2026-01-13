#!/bin/bash
#SBATCH --account=qingjiem-heart-tte
#SBATCH --qos=bham
#SBATCH --time=12:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:2
#SBATCH --gpus-per-task 2
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_80
#SBATCH --mem=32G  # 请求内存
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

# accelerate launch \
#     --multi_gpu \
#     --mixed_precision=no \
#     --main_process_port=0 \
#     eval.py \
#     --config=configs_shift/acdc_wholeheart_uncond_uvit_greyscale_mid_4by4.py \
#     --workdir=output_shift/acdc_wholeheart_uncond_greyscale_uvit_mid_Y_bound_4by4

# accelerate launch \
#     --multi_gpu \
#     --mixed_precision=no \
#     --main_process_port=0 \
#     eval.py \
#     --config=configs_shift/acdc_wholeheart_uncond_uvit_MoE_greyscale_mid_4by4.py \
#     --workdir=output_shift/acdc_wholeheart_uncond_MoE_greyscale_uvit_mid_4by4_Y_bound


# accelerate launch \
#     --multi_gpu \
#     --mixed_precision=no \
#     --main_process_port=0 \
#     eval.py \
#     --config=configs/acdc_wholeheart_uncond_uvit_greyscale_mid_4by4.py \
#     --workdir=output/acdc_wholeheart_uncond_greyscale_uvit_mid_4by4_old


# accelerate launch \
#     --multi_gpu \
#     --mixed_precision=no \
#     --main_process_port=0 \
#     eval.py \
#     --config=configs_shift/echonet_dynamic_uncond_uvit_greyscale_mid_4by4.py \
#     --workdir=output_shift/echonet_dynamic_uncond_greyscale_uvit_mid_Y_bound    

accelerate launch \
    --multi_gpu \
    --mixed_precision=no \
    --main_process_port=0 \
    eval.py \
    --config=configs_shift/echonet_dynamic_uncond_uvit_MoE_greyscale_mid_4by4.py \
    --workdir=output_shift/echonet_dynamic_uncond_MoE_greyscale_uvit_mid_Y_bound 