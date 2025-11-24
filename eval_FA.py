import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from tools.fid_score import calculate_fid_given_paths
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


def calculate_inception_score(sample_dir, batch_size=32, splits=10, device=None):
    """
    计算 Inception Score
    
    Args:
        sample_dir: 样本图像目录
        batch_size: 批处理大小
        splits: 用于计算 IS 的分割数
    
    Returns:
        is_mean: IS 平均值
        is_std: IS 标准差
    """
    try:
        from torchvision.models import inception_v3
    except ImportError:
        logging.error("torchvision not installed. Please install it to calculate IS.")
        return 0.0, 0.0
    
    device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 加载预训练的 InceptionV3 模型
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # 移除分类层
    inception_model.fc = nn.Identity()
    
    # 加载样本图像
    sample_files = sorted(glob.glob(os.path.join(sample_dir, '*.png')))
    if not sample_files:
        sample_files = sorted(glob.glob(os.path.join(sample_dir, '*.jpg')))
    
    if not sample_files:
        logging.warning(f"No image files found in {sample_dir}")
        return 0.0, 0.0
    
    logging.info(f"Found {len(sample_files)} samples for IS calculation")
    
    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 提取特征
    features_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sample_files), batch_size), desc="Extracting features"):
            batch_files = sample_files[i:i+batch_size]
            batch_images = []
            
            for img_file in batch_files:
                try:
                    img = Image.open(img_file).convert('RGB')
                    img_tensor = preprocess(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    logging.warning(f"Failed to load image {img_file}: {e}")
                    continue
            
            if batch_images:
                batch_images = torch.stack(batch_images).to(device)
                features = inception_model(batch_images).cpu().numpy()
                features_list.append(features)
    
    if not features_list:
        logging.warning("No valid images found for IS calculation")
        return 0.0, 0.0
    
    # 合并所有特征
    all_features = np.concatenate(features_list, axis=0)
    logging.info(f"Total features shape: {all_features.shape}")
    
    # 计算 IS
    is_scores = []
    
    n_splits = splits
    split_size = len(all_features) // n_splits
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else len(all_features)
        
        split_features = all_features[start_idx:end_idx]
        
        # 对特征进行 softmax
        p_y = np.mean(split_features, axis=0)
        
        # 计算条件熵
        p_yx = split_features / np.sum(split_features, axis=1, keepdims=True)
        
        # 避免 log(0)
        p_yx = np.clip(p_yx, 1e-10, 1.0)
        
        # 计算 IS
        kl_divergence = np.sum(p_yx * (np.log(p_yx) - np.log(p_y)), axis=1)
        is_score = np.exp(np.mean(kl_divergence))
        is_scores.append(is_score)
    
    is_scores = np.array(is_scores)
    is_mean = float(np.mean(is_scores))
    is_std = float(np.std(is_scores))
    
    logging.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    return is_mean, is_std


def calculate_lpips_score(sample_dir, real_dir, device, batch_size=32):
    """计算 LPIPS 分数"""
    lpips_model = LPIPS(net='alex', version='0.1').to(device).eval()
    # 加载生成的样本
    sample_files = sorted(glob.glob(os.path.join(sample_dir, '*.png')))
    if not sample_files:
        sample_files = sorted(glob.glob(os.path.join(sample_dir, '*.jpg')))
    
    # 加载真实的样本
    real_files = sorted(glob.glob(os.path.join(real_dir, '*.png')))
    if not real_files:
        real_files = sorted(glob.glob(os.path.join(real_dir, '*.jpg')))
    
    if not sample_files:
        logging.warning(f"No samples found in {sample_dir}")
        return 0.0
    
    lpips_scores = []
    
    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sample_files), batch_size), desc="Computing LPIPS"):
            batch_files = sample_files[i:i+batch_size]
            batch_real_files = real_files[i:i+batch_size]
            
            # 加载图像
            images1 = []
            images2 = []
            
            for img_file, real_img_file in zip(batch_files, batch_real_files):
                try:
                    img = Image.open(img_file).convert('RGB')
                    img_tensor = preprocess(img).to(device)
                    images1.append(img_tensor)
                    
                    # 对于无配对图像的情况，计算与略微扰动版本的 LPIPS
                    # 或加载对应的真实图像
                    real_img = Image.open(real_img_file).convert('RGB')
                    real_tensor = preprocess(real_img).to(device)
                    images2.append(real_tensor)
                except Exception as e:
                    logging.warning(f"Failed to load image {img_file}: {e}")
                    continue
            
            if images1 and images2 and len(images1) == len(images2):
                images1 = torch.stack(images1)
                images2 = torch.stack(images2)
                
                # 计算 LPIPS
                lpips_batch = lpips_model(images1, images2)
                lpips_scores.extend(lpips_batch.cpu().numpy().flatten().tolist())
    
    if lpips_scores:
        lpips_mean = float(np.mean(lpips_scores))
        lpips_std = float(np.std(lpips_scores))
        logging.info(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")
        return lpips_mean
    else:
        logging.warning("No valid image pairs found for LPIPS calculation")
        return 0.0


def eval_checkpoints(config):
    """评估保存的检查点"""
    
    # 初始化加速器
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    device = accelerator.device
    accelerator.print(f"Accelerator initialized. Using device: {device}")
    
    # 加载数据集
    dataset = get_dataset(**config.dataset)
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
                utils.PositionalTokenSample2dir(
                    accelerator,
                    sample_dir,
                    samples_per_process,
                    config.eval.get('mini_batch_size', 50),
                    sample_fn,
                    img_sz=config.dataset.resolution,
                    low_freqs=config.dataset.low_freqs,
                    block_sz=config.dataset.block_sz,
                    denorm_fn=train_dataset.denormalize if hasattr(train_dataset, 'denormalize') else None,
                    reverse_ordering_fn=train_dataset.reverse_ordering if hasattr(train_dataset, 'reverse_ordering') else None
                )
            
            # 确保所有进程完成
            accelerator.wait_for_everyone()
            
            # 计算评估指标（仅主进程）
            if accelerator.is_main_process:
                logging.info("Computing evaluation metrics...")
                
                # 计算 FID
                # fid_score = calculate_fid_given_paths(
                #     (dataset.fid_stat, sample_dir)
                # )
                
                # 计算 IS
                inception_score, inception_score_std = calculate_inception_score(
                    sample_dir,
                    batch_size=config.eval.get('is_batch_size', 32),
                    splits=10
                )
                
                # 计算 LPIPS
                lpips_score = calculate_lpips_score(
                    sample_dir,
                    '/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs',  # 真实图像目录
                    device,
                    batch_size=config.eval.get('lpips_batch_size', 32)
                )
                
                result = {
                    'step': step,
                    # 'fid': fid_score,
                    'is_mean': inception_score,
                    'is_std': inception_score_std,
                    'lpips': lpips_score
                }
                results.append(result)
                
                logging.info(f"Step {step} Results:")
                # logging.info(f"  FID: {fid_score:.4f}")
                logging.info(f"  IS: {inception_score:.4f} ± {inception_score_std:.4f}")
                logging.info(f"  LPIPS: {lpips_score:.4f}")
                
                # 保存单个结果
                result_file = os.path.join(results_dir, f'metrics_step_{step}.txt')
                with open(result_file, 'w') as f:
                    f.write(f"Step: {step}\n")
                    # f.write(f"FID: {fid_score:.6f}\n")
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
            f.write("Step\tIS_mean\tIS_std\tLPIPS\n")
            for result in results:
                f.write(f"{result['step']}\t{result['is_mean']:.6f}\t{result['is_std']:.6f}\t{result['lpips']:.6f}\n")
        
        logging.info(f"\nAll results saved to {results_file}")
        
        # 打印总结
        logging.info("\n" + "="*50)
        logging.info("Evaluation Summary")
        logging.info("="*50)
        for result in results:
            logging.info(f"Step {result['step']}: IS={result['is_mean']:.4f}±{result['is_std']:.4f}, LPIPS={result['lpips']:.4f}")


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
