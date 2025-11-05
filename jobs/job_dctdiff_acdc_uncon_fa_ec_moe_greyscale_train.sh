#!/bin/bash
#SBATCH --account=chenhp-data-gen
#SBATCH --qos=bham
#SBATCH --time=128:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:2
#SBATCH --gpus-per-task 2
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_40
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
# accelerate test
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r8.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l8r8
# nohup python train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r8.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l8r8

# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r16.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l8r16
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r12.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l8r12_minmax
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l6r9.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l6r9_minmax
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l6r16.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l6r16_minmax
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l6r12.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l6r12_minmax
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l4r9.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l4r9_minmax
# accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l5r9.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l5r9_minmax
accelerate launch --multi_gpu --mixed_precision=no train_greyscale_FA_MoE.py --config=configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l4r9_low8.py --workdir output/acdc_wholeheart_uncond_fa_ec_moe_greyscale_uvit_mid_4by4_l4r9_low8_minmax