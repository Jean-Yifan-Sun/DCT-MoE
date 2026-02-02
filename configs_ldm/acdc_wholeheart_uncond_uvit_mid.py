import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    name = 'acdc_wholeheart_uncond_uvit_mid'
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
        name='uvit',  # use greyscale UViT
        tokens=144,  # number of tokens to the network
        in_chans=16,  # B**2 - m
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
    
    config.z_shape = [12 * 12, 16]  # latent shape (channels, height, width)
    config.latent_shape = [16, 12, 12] 
    
    config.dataset = d(
        name='acdc_uncond_images',
        path='data/scratch/datasets/ACDC/Unlabeled/Wholeheart',
        resolution=96
    )

    config.sample = d(
        save_start=100000,
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
