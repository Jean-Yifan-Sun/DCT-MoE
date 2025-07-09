#!/bin/bash
#SBATCH --account=qingjiem-heart-tte
#SBATCH --qos=bham
#SBATCH --time=1:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --gpus-per-task 1
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
# accelerate test
python -m pytorch_fid --save-stats /bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/ACDC/Image  /bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/U-ViT2/assets/fid_stats/acdc_labeled_greyscale.npz
nohup python -m pytorch_fid data/scratch/datasets/ACDC/Image data/scratch/datasets/ACDC/recon_acdc_cond_coe16