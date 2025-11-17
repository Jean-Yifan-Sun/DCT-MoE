import numpy as np
from datasets import DCT_FA_Customized
import os,torch
from tqdm import tqdm
import ml_collections
import matplotlib.pyplot as plt

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def cache_dataset(dataset, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset))):
        # 获取 DCT tokens (已经是 torch.Tensor)
        tokens = dataset[idx]
        # 确保数据类型一致
        tokens = tokens.float()  # 使用 float32
        
        # 直接用 torch.save 保存
        save_path = os.path.join(cache_dir, f"{idx}.pt")
        torch.save(tokens, save_path)
        
        # 立即验证
        loaded = torch.load(save_path, weights_only=True)
        if not torch.allclose(tokens, loaded, rtol=1e-5, atol=1e-8):
            print("警告：样本的保存验证失败")
            diff = torch.abs(tokens - loaded)
            print(f"最大差异: {diff.max().item()}")

def verify_cache(original_dataset, cache_dir):
    for idx in tqdm(range(len(original_dataset))):
        cache_path = os.path.join(cache_dir, f"{idx}.pt")
        assert os.path.exists(cache_path), f"Missing cache for index {idx}"
        
        # 获取原始数据和缓存数据
        original = original_dataset[idx]
        cached = torch.load(cache_path, weights_only=True)
        
        # 详细的比较信息
        if not torch.allclose(original, cached, rtol=1e-5, atol=1e-8):
            diff = torch.abs(original - cached)
            print(f"验证失败，索引: {idx}")
            print(f"最大差异: {diff.max().item()}")
            print(f"平均差异: {diff.mean().item()}")
            print(f"原始张量范围: [{original.min().item()}, {original.max().item()}]")
            print(f"缓存张量范围: [{cached.min().item()}, {cached.max().item()}]")
            print(f"原始张量形状: {original.shape}")
            print(f"缓存张量形状: {cached.shape}")
            raise AssertionError("张量不匹配")

def plot_distribution(original_dataset, cache_dir):
    

    original_values = []
    cached_values = []

    for idx in range(len(original_dataset)):
        original = original_dataset[idx].flatten().numpy()
        cached = torch.load(os.path.join(cache_dir, f"{idx}.pt"), weights_only=True).flatten().numpy()

        original_values.extend(original)
        cached_values.extend(cached)

    plt.figure(figsize=(12, 6))
    plt.hist(original_values, bins=100, alpha=0.5, label='Original', density=True)
    plt.hist(cached_values, bins=100, alpha=0.5, label='Cached', density=True)
    plt.legend()
    plt.title('Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(os.path.join(cache_dir, 'distribution_comparison.png'))
    plt.close()

if __name__ == "__main__":
    # 示例用法
    config = ml_collections.ConfigDict()

    num_fa_length = 4  # number of frequency aware coefficients length
    num_fa_repeats_x = 4  # number of frequency aware repeats for each length in x direction
    num_fa_repeats_y = 4  # number of frequency aware repeats for each length in y direction
    num_fa_repeats = num_fa_repeats_x * num_fa_repeats_y  # number of frequency aware repeats for each length
    low_freqs = 12  # B**2 - m
    block_sz = 4  # B
    normalization = "minmax"  # 归一化方法

    config.dataset = d(
        name='acdc_uncond',
        path='data/scratch/datasets/ACDC/Unlabeled/Wholeheart',
        dataset_type='wholeheart',  # use wholeheart dataset
        resolution=96,
        tokens=int(low_freqs*96*96/(num_fa_repeats * num_fa_length * block_sz**2)),  # number of tokens to the network
        low_freqs=low_freqs,  # B**2 - m
        block_sz=block_sz,  # B
        Y_bound=[774.0],  # eta
        Y_mean=[241.045, -0.725, 0.44, 0.107, 0.009, 0.154, -0.057, 0.0, -0.008, 0.019, 0.009, -0.012, 0.005, 0.002, 0.004, -0.002],  # eta
        Y_std=[182.467, 32.821, 34.996, 14.001, 17.621, 13.189, 5.79, 9.609, 9.805, 5.652, 4.634, 6.502, 4.963, 3.796, 3.44, 2.178],
        Y_min=[9.5, -119.312, -122.375, -51.25, -58.906, -49.5, -20.688, -32.906, -33.094, -19.891, -15.891, -21.75, -16.969, -12.773, -11.648, -7.492],
        Y_max=[774.0, 112.375, 125.812, 49.0, 58.5, 46.5, 20.344, 32.906, 33.406, 19.953, 15.953, 22.0, 16.984, 12.812, 11.68, 7.461],
        Y_entropy=[5.287, 2.936, 3.005, 1.94, 2.216, 1.866, 1.186, 1.561, 1.581, 1.169, 1.024, 1.252, 1.071, 1.0, 1.0, 1.0],
        SNR_scale=4.0,
        greyscale=True,  # use greyscale images
        reweight=True,  # use loss reweighting based on entropy
        tempature=1.0,  # temperature for loss reweighting
        reweight_dim=-1,  # dimension to apply loss reweighting (1: channel-wise, 2: token-wise, -1: element-wise)
        frequency_aware_tokens=True,  # use frequency aware tokens
        tokenwise_normalization=normalization,
        num_fa_length=num_fa_length,  # number of frequency aware coefficients length
        num_fa_repeats=num_fa_repeats,  # number of frequency aware repeats for each length
        num_fa_repeats_x=num_fa_repeats_x,  # number of frequency aware repeats for each length in x direction
        num_fa_repeats_y=num_fa_repeats_y,  # number of frequency aware repeats for each length in y direction
    )
    
    kwargs = config.dataset.to_dict()
    path = kwargs.get('path', '')
    train = DCT_FA_Customized(
                    data_property={'mean': kwargs.get('Y_mean', None), 'std': kwargs.get('Y_std', None),'min': kwargs.get('Y_min', None), 'max': kwargs.get('Y_max', None)},
                    **kwargs
                )

    cache_dir = os.path.join(kwargs.get('path'), f'cache_dct_fa_{block_sz}by{block_sz}_low{low_freqs}_l{num_fa_length}r{num_fa_repeats}_{normalization}')
    cache_dataset(train, cache_dir)

    # 可选:验证缓存
    # verify_cache(train, cache_dir)

    # 可选:绘制分布图
    # plot_distribution(train, cache_dir)