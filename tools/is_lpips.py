import torch
import torch.nn as nn
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from lpips import LPIPS
from absl import logging

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

