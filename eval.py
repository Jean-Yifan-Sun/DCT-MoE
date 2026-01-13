import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from tools.fid_score import calculate_fid_given_paths
from tools.is_lpips import calculate_lpips_score, calculate_inception_score
from lpips import LPIPS
from absl import logging
from datetime import timedelta
from accelerate import InitProcessGroupKwargs, Accelerator
from DCT_utils import zigzag_order, reverse_zigzag_order
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import utils
import sde as sde_lib
from ml_collections import config_flags
from absl import flags, app
import glob
import shutil
from PIL import Image
from torchvision import transforms
from scipy.stats import entropy

def eval_checkpoints(config):
    """评估保存的检查点"""
    
    # 初始化加速器
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=20000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    device = accelerator.device
    accelerator.print(f"Accelerator initialized. Using device: {device}")
    
    # 加载数据集
    dataset = get_dataset(**config.dataset)
    greyscale = config.dataset.get('greyscale', False)
    train_dataset = dataset.get_split(split='train', labeled=(config.train.mode == 'cond'))
    accelerator.print(f'Data shape: {dataset.data_shape}')
    
    # 加载模型
    nnet = utils.get_nnet(**config.nnet).to(device)
    nnet_ema = utils.get_nnet(**config.nnet).to(device)
    nnet_ema.eval()
    
    score_model = sde_lib.ScoreModel(
        nnet_ema,
        pred=config.pred,
        sde=sde_lib.VPSDE(SNR_scale=config.dataset.SNR_scale)
    )
    
    # 初始化 LPIPS 计算器
    # lpips_model = LPIPS(net='alex', version='0.1').to(device).eval()
    
    # 获取所有检查点
    ckpt_root = config.ckpt_root
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_root, '*.ckpt')),
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    eval_start = config.eval.get('eval_start', 0)
    ckpts = [ckpt for ckpt in ckpts if int(os.path.basename(ckpt).split('.')[0]) >= eval_start]
    
    if accelerator.is_main_process:
        logging.info(f"Found {len(ckpts)} checkpoints")
    
    # 准备结果保存目录
    results_dir = os.path.join(config.eval_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    # 遍历每个检查点
    for ckpt_path in ckpts:
        step = int(os.path.basename(ckpt_path).split('.')[0])
        
        if accelerator.is_main_process:
            logging.info(f"\n{'='*50}")
            logging.info(f"Evaluating checkpoint: {step}")
            logging.info(f"{'='*50}")
        
        try:
            # 加载检查点
            train_state = utils.TrainState(
                optimizer=None, 
                lr_scheduler=None,
                step=step,
                nnet=nnet, 
                nnet_ema=nnet_ema)
            train_state.resume(ckpt_root, step=step)
            
            accelerator.wait_for_everyone()
            
            # 生成样本
            sample_dir = os.path.join(config.eval_dir, f'samples_step_{step}')
            os.makedirs(sample_dir, exist_ok=True)
            
            n_samples = config.eval.get('n_samples', 50000)
            samples_per_process = n_samples
            
            if accelerator.is_main_process:
                logging.info(f"Generating {n_samples} samples...")
            
            with torch.no_grad():
                # 定义采样函数
                def sample_fn(batch_size):
                    _x_init = torch.randn(
                        batch_size,
                        *dataset.data_shape,
                        device=device
                    )
                    kwargs = dict(use_moe=config.nnet.use_moe)
                    
                    noise_schedule = NoiseScheduleVP(
                        schedule='linear',
                        SNR_scale=config.dataset.SNR_scale
                    )
                    model_fn = model_wrapper(
                        score_model.noise_pred,
                        noise_schedule,
                        time_input_type='0',
                        model_kwargs=kwargs
                    )
                    dpm_solver = DPM_Solver(model_fn, noise_schedule)
                    samples = dpm_solver.sample(
                        _x_init,
                        steps=config.eval.get('sample_steps', 100),
                        eps=1e-4,
                        adaptive_step_size=False,
                        fast_version=True
                    )
                    return samples
                
                # 生成并保存样本
                if greyscale:
                    utils.DCTsample2dir_greyscale(
                        accelerator,
                        sample_dir,
                        samples_per_process,
                        config.eval.get('mini_batch_size', 50),
                        sample_fn,
                        tokens=config.nnet.tokens,
                        resolution=config.dataset.resolution,
                        low_freqs=config.dataset.low_freqs,
                        block_sz=config.dataset.block_sz,
                        reverse_order=reverse_zigzag_order(config.dataset.block_sz),
                        Y_bound=config.dataset.Y_bound
                    )
                else:
                    utils.DCTsample2dir(
                        accelerator,
                        sample_dir,
                        samples_per_process,
                        config.eval.get('mini_batch_size', 50),
                        sample_fn,
                        tokens=config.nnet.tokens,
                        resolution=config.dataset.resolution,
                        low_freqs=config.dataset.low_freqs,
                        block_sz=config.dataset.block_sz,
                        reverse_order=reverse_zigzag_order(config.dataset.block_sz),
                        Y_bound=config.dataset.Y_bound
                    )
            
            # 确保所有进程完成
            accelerator.wait_for_everyone()
            
            # 计算评估指标（仅主进程）
            if accelerator.is_main_process:
                logging.info("Computing evaluation metrics...")
                
                # 计算 FID
                fid_score = calculate_fid_given_paths(
                    (dataset.fid_stat, sample_dir)
                )
                
                # 计算 IS
                inception_score, inception_score_std = calculate_inception_score(
                    sample_dir,
                    batch_size=config.eval.get('is_batch_size', 32),
                    splits=10
                )
                
                # 计算 LPIPS
                real_samples_dir = '/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/EchoNet-Dynamic/images'
                # real_samples_dir = '/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs'
                lpips_score = calculate_lpips_score(
                    sample_dir,
                    real_samples_dir,  # 真实图像目录
                    device,
                    batch_size=config.eval.get('lpips_batch_size', 32)
                )
                
                result = {
                    'step': step,
                    'fid': fid_score,
                    'is_mean': inception_score,
                    'is_std': inception_score_std,
                    'lpips': lpips_score
                }
                results.append(result)
                
                logging.info(f"Step {step} Results:")
                logging.info(f"  FID: {fid_score:.4f}")
                logging.info(f"  IS: {inception_score:.4f} ± {inception_score_std:.4f}")
                logging.info(f"  LPIPS: {lpips_score:.4f}")
                
                # 保存单个结果
                result_file = os.path.join(results_dir, f'metrics_step_{step}.txt')
                with open(result_file, 'w') as f:
                    f.write(f"Step: {step}\n")
                    f.write(f"FID: {fid_score:.6f}\n")
                    f.write(f"IS (mean): {inception_score:.6f}\n")
                    f.write(f"IS (std): {inception_score_std:.6f}\n")
                    f.write(f"LPIPS: {lpips_score:.6f}\n")
                
                # 清理样本目录以节省空间
                if config.eval.get('cleanup_samples', True):
                    shutil.rmtree(sample_dir)
                    logging.info(f"Cleaned up samples directory: {sample_dir}")
            
            accelerator.wait_for_everyone()
            
        except Exception as e:
            if accelerator.is_main_process:
                logging.error(f"Error evaluating checkpoint {step}: {str(e)}")
            accelerator.wait_for_everyone()
            continue
    
    # 保存所有结果
    if accelerator.is_main_process:
        results_file = os.path.join(results_dir, 'all_metrics.txt')
        with open(results_file, 'w') as f:
            f.write("Step\tFID\tIS_mean\tIS_std\tLPIPS\n")
            for result in results:
                f.write(f"{result['step']}\t{result['fid']:.6f}\t{result['is_mean']:.6f}\t{result['is_std']:.6f}\t{result['lpips']:.6f}\n")
        
        logging.info(f"\nAll results saved to {results_file}")
        
        # 打印总结
        logging.info("\n" + "="*50)
        logging.info("Evaluation Summary")
        logging.info("="*50)
        for result in results:
            logging.info(f"Step {result['step']}: FID={result['fid']:.4f}, IS={result['is_mean']:.4f}±{result['is_std']:.4f}, LPIPS={result['lpips']:.4f}")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "Evaluation configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    flags.DEFINE_string("workdir", None, "Work unit directory.")
    
    def main(argv):
        config = FLAGS.config
        config.workdir = FLAGS.workdir or 'exp_train_pos_token'
        config.ckpt_root = os.path.join(config.workdir, 'ckpts')
        eval_checkpoints(config)
    
    app.run(main)
