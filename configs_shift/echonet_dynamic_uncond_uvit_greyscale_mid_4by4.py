import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    name = 'echonet_dynamic_uncond_uvit_greyscale_mid_4by4'
    config.seed = 1234
    config.pred = 'noise_pred'
    config.name = name
    config.eval_dir = f"output_shift/evaluation/{name}"
    config.eval = d(
        eval_start=100000,
        n_samples=2000,
        mini_batch_size=500,
        sample_steps=100,
        is_batch_size=32,
        lpips_batch_size=32,
        cleanup_samples=True,
    )

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

    config.nnet = d(
        name='uvit_greyscale',  # use greyscale UViT
        tokens=196,  # number of tokens to the network
        low_freqs=12,  # B**2 - m
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=True,
        num_classes=-1,
        use_moe=False,
    )

    config.dataset = d(
        name='echonet',
        path='data/scratch/datasets/EchoNet-Dynamic',
        dataset_type='dynamic',  # use wholeheart dataset
        resolution=112,
        tokens=196,  # number of tokens to the network
        low_freqs=12,  # B**2 - m
        block_sz=4,  # B
        Y_bound=[512.0],  # eta
        Y_mean=[-390.081, 0.103, -0.947, 0.171, 0.192, 0.203, 0.001, 0.061, 0.233, -0.075, -0.051, -0.199, -0.046, -0.076, -0.042, 0.029],  # eta
        Y_std=[163.706, 28.348, 26.485, 11.872, 13.871, 8.085, 2.919, 6.039, 8.605, 7.49, 5.683, 4.875, 2.372, 2.172, 3.337, 1.668],
        Y_min=[-512.0, -108.688, -112.5, -52.5, -55.75, -36.75, -14.586, -27.062, -37.156, -36.156, -28.484, -27.75, -13.672, -14.266, -19.562, -10.367],
        Y_max=[148.0, 107.812, 89.625, 47.0, 56.938, 34.0, 14.758, 29.281, 38.125, 35.156, 27.938, 22.0, 12.641, 12.125, 18.391, 11.641],
        Y_entropy=[5.419, 2.769, 2.659, 1.88, 2.022, 1.575, 1.018, 1.338, 1.593, 1.484, 1.309, 1.163, 0.928, 0.925, 1.051, 0.852],
        SNR_scale=4.0,
        greyscale=True,  # use greyscale images
        reweight=True,  # use loss reweighting based on entropy
        temperature=0.0,  # temperature for loss reweighting
        reweight_dim=1,
        cache=True,  # cache dataset in memory
        cache_name='cached_echonet_dynamic_4by4_4Y'
    )

    config.sample = d(
        save_start=10000,
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
