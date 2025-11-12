import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.name = 'acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4'

    num_fa_length = 8  # number of frequency aware coefficients length
    num_fa_repeats = 16  # number of frequency aware repeats for each length
    num_fa_repeats_x = 4  # number of frequency aware repeats for each length in x direction
    num_fa_repeats_y = 4  # number of frequency aware repeats for
    low_freqs = 16  # B**2 - m
    block_sz = 4  # B
    
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
        name='uvit_greyscale_moe',  # use greyscale UViT
        tokens=int(low_freqs*96*96/(num_fa_repeats * num_fa_length * block_sz**2)),  # number of tokens to the network
        low_freqs=low_freqs,  # B**2 - m
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=True,
        num_classes=-1,
        in_chans=num_fa_repeats * num_fa_length,
        use_moe=True,
        MoE={
            "type": "ecmoe",  # 'normal', 'ecmoe'
            "depth": 1,
            "num_experts": 4,
            "router": "topk",
            "top_k": 2,
            "noise_eps": 1e-2,
            "aux_loss_alpha": 0.00
        },

    )

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
        tokenwise_normalization="minmax",
        num_fa_length=num_fa_length,  # number of frequency aware coefficients length
        num_fa_repeats=num_fa_repeats,  # number of frequency aware repeats for each length
        num_fa_repeats_x=num_fa_repeats_x,  # number of frequency aware repeats for each length in x direction
        num_fa_repeats_y=num_fa_repeats_y,  # number of frequency aware repeats for each length in y direction
    )

    config.sample = d(
        save_start=20000,
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
