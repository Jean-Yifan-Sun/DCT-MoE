import numpy as np
from datasets import DCT_FA_Customized, DCT_4Y
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

    num_fa_length = 8  # number of frequency aware coefficients length
    num_fa_repeats_x = 4  # number of frequency aware repeats for each length in x direction
    num_fa_repeats_y = 4  # number of frequency aware repeats for each length in y direction
    num_fa_repeats = num_fa_repeats_x * num_fa_repeats_y  # number of frequency aware repeats for each length
    low_freqs = 16  # B**2 - m
    block_sz = 4  # B
    normalization = "Y_bound"  # 归一化方法

    config.dataset = d(
        name='acdc_uncond',
        path='data/scratch/datasets/ACDC/Unlabeled/Wholeheart',
        dataset_type='wholeheart',  # use wholeheart dataset
        resolution=96,
        # tokens=int(low_freqs*96*96/(num_fa_repeats * num_fa_length * block_sz**2)),  # number of tokens to the network
        tokens=144,  # number of tokens to the network 4Y
        low_freqs=low_freqs,  # B**2 - m
        block_sz=block_sz,  # B
        Y_bound=[502.5],  # eta
        Y_mean=[-270.982, -0.725, 0.44, 0.107, 0.009, 0.154, -0.058, 0.0, -0.008, 0.019, 0.009, -0.012, 0.005, 0.002, 0.004, -0.002],  # eta
        Y_std=[182.433, 32.822, 34.997, 14.001, 17.621, 13.189, 5.791, 9.609, 9.805, 5.653, 4.634, 6.503, 4.964, 3.796, 3.44, 2.178],
        Y_min=[-502.5, -119.312, -122.375, -51.25, -58.906, -49.5, -20.703, -32.906, -33.094, -19.891, -15.891, -21.75, -16.969, -12.773, -11.648, -7.492],
        Y_max=[262.0, 112.375, 125.812, 49.0, 58.5, 46.5, 20.344, 32.906, 33.406, 19.953, 15.953, 22.0, 16.984, 12.812, 11.68, 7.461],
        Y_entropy=[5.883, 3.501, 3.575, 2.418, 2.75, 2.341, 1.479, 2.017, 2.044, 1.453, 1.316, 1.586, 1.365, 1.174, 1.108, 1.0],
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

    # config.dataset = d(
    #     name='echonet',
    #     path='data/scratch/datasets/EchoNet-Dynamic',
    #     dataset_type='dynamic',  # use wholeheart dataset
    #     resolution=112,
    #     tokens=int(low_freqs*112*112/(num_fa_repeats * num_fa_length * block_sz**2)),
    #     # tokens=196,  # number of tokens to the network
    #     low_freqs=low_freqs,  # B**2 - m
    #     block_sz=block_sz,  # B
    #     Y_bound=[512.0],  # eta
    #     Y_mean=[-390.081, 0.103, -0.947, 0.171, 0.192, 0.203, 0.001, 0.061, 0.233, -0.075, -0.051, -0.199, -0.046, -0.076, -0.042, 0.029],  # eta
    #     Y_std=[163.706, 28.348, 26.485, 11.872, 13.871, 8.085, 2.919, 6.039, 8.605, 7.49, 5.683, 4.875, 2.372, 2.172, 3.337, 1.668],
    #     Y_min=[-512.0, -108.688, -112.5, -52.5, -55.75, -36.75, -14.586, -27.062, -37.156, -36.156, -28.484, -27.75, -13.672, -14.266, -19.562, -10.367],
    #     Y_max=[148.0, 107.812, 89.625, 47.0, 56.938, 34.0, 14.758, 29.281, 38.125, 35.156, 27.938, 22.0, 12.641, 12.125, 18.391, 11.641],
    #     Y_entropy=[5.419, 2.769, 2.659, 1.88, 2.022, 1.575, 1.018, 1.338, 1.593, 1.484, 1.309, 1.163, 0.928, 0.925, 1.051, 0.852],
    #     SNR_scale=4.0,
    #     greyscale=True,  # use greyscale images
    #     reweight=True,  # use loss reweighting based on entropy
    #     tempature=1.0,  # temperature for loss reweighting
    #     reweight_dim=-1,
    #     frequency_aware_tokens=True,  # use frequency aware tokens
    #     tokenwise_normalization=normalization,
    #     num_fa_length=num_fa_length,  # number of frequency aware coefficients length
    #     num_fa_repeats=num_fa_repeats,  # number of frequency aware repeats for each length
    #     num_fa_repeats_x=num_fa_repeats_x,  # number of frequency aware repeats for each length in x direction
    #     num_fa_repeats_y=num_fa_repeats_y,  # number of frequency aware repeats for each length in y direction
    # )
    
    kwargs = config.dataset.to_dict()
    path = kwargs.get('path', '')
    img_sz = kwargs.get('resolution')
    train = DCT_FA_Customized(
                    data_property={'mean': kwargs.get('Y_mean', None), 'std': kwargs.get('Y_std', None),'min': kwargs.get('Y_min', None), 'max': kwargs.get('Y_max', None), 'Y_bound': kwargs.get('Y_bound', None)},
                    img_sz=img_sz,
                    **kwargs
                )
    # train = DCT_4Y(
    #     path=path,
    #     img_sz=kwargs.get('resolution'),
    #     tokens=kwargs.get('tokens'),
    #     low_freqs=kwargs.get('low_freqs'),
    #     block_sz=kwargs.get('block_sz'),
    #     Y_bound=kwargs.get('Y_bound'),
    #     cache=False,
    #     cache_name=f'acdc_uncond_wholeheart_4Y_low{low_freqs}',
    # )

    cache_dir = os.path.join(kwargs.get('path'), f'cache_dct_fa_{block_sz}by{block_sz}_low{low_freqs}_l{num_fa_length}r{num_fa_repeats}_{normalization}')
    # cache_dir = os.path.join(kwargs.get('path'), f'cache_echo_dct_fa_{block_sz}by{block_sz}_low{low_freqs}_l{num_fa_length}r{num_fa_repeats}_{normalization}')
    # cache_dir = os.path.join(kwargs.get('path'), f'acdc_uncond_wholeheart_4Y_low{low_freqs}')
    cache_dataset(train, cache_dir)
    # 可选:验证缓存
    # verify_cache(train, cache_dir)

    # 可选:绘制分布图
    # plot_distribution(train, cache_dir)