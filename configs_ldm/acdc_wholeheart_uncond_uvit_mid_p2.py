import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def compute_z_shape(latent_shape, patch_size):
    """
    根据 latent_shape 和 patch_size 自动计算 z_shape
    
    Args:
        latent_shape: [C, H, W] - 自编码器输出的潜在空间形状
        patch_size: patch 大小（1 表示不使用 patchify）
    
    Returns:
        z_shape: [num_tokens, token_dim]
    
    公式:
        num_tokens = (H / patch_size) * (W / patch_size)
        token_dim = C * patch_size^2
    """
    C, H, W = latent_shape
    
    if patch_size == 1 or patch_size is None:
        # 不使用 patchify
        num_tokens = H * W
        token_dim = C
    else:
        # 使用 patchify
        num_tokens = (H // patch_size) * (W // patch_size)
        token_dim = C * (patch_size ** 2)
    
    return [num_tokens, token_dim]


def get_config():
    config = ml_collections.ConfigDict()

    name = 'acdc_wholeheart_uncond_uvit_mid'
    config.seed = 1234
    config.pred = 'noise_pred'
    config.name = name

    config.train = d(
        n_steps=500000,
        batch_size=512,
        mode='uncond',
        log_interval=100,
        eval_interval=25000,
        save_interval=25000,
    )
    
    config.private = d(
        use_dp=False,
        dp_method='dpsgd',
        accountant='prv',
        secure_mode=False,  # use secure mode for DP training
        target_epsilon=10,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.5,  # Adjusted for DP training
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.encode = d(
        patch_size=2,  # ← 只需改这一个地方！1 或者 null 表示不使用 patchify，其他值表示使用对应的 patch 大小
    )

    # 原始潜在空间形状（自编码器输出）
    config.latent_shape = [16, 12, 12]  # [C, H, W]
    
    # 自动计算 z_shape（不需要手动改）
    config.z_shape = compute_z_shape(config.latent_shape, config.encode.patch_size)

    config.nnet = d(
        name='uvit',  # use greyscale UViT
        tokens=config.z_shape[0],  # number of tokens to the network
        in_chans=config.z_shape[1],  # B**2 - m
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=True,
        num_classes=-1,
        use_moe=False,
    )

    config.autoencoder = d(
        pretrained_path='black-forest-labs/FLUX.1-dev'  # or leave as-is if using diffusers
    )
    
    config.dataset = d(
        name='acdc_uncond_images',
        path='data/scratch/datasets/ACDC/Unlabeled/Wholeheart',
        resolution=96,
        SNR_scale=1
    )

    config.sample = d(
        save_start=100000,
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        # path='data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    config.eval_dir = f"output_ldm/acdc_wholeheart_uncond_uvit_mid_p2"
    config.eval = d(
        eval_start=100000,
        lpips_n_samples=2000,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        sample_steps=50,
        is_batch_size=32,
        lpips_batch_size=32,
        cleanup_samples=True,
        real_dir='data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs',  # 用于计算 LPIPS 的真实图像目录
    )

    return config
