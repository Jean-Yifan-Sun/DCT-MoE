import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.name = 'acdc_wholeheart_uncond_uvit_FA_EC_6MoE8_greyscale_mid_4by4_low16'

    num_fa_length = 8  # number of frequency aware coefficients length
    num_fa_repeats_x = 3  # number of frequency aware repeats for each length in x direction
    num_fa_repeats_y = 3  # number of frequency aware repeats for
    num_fa_repeats = num_fa_repeats_x * num_fa_repeats_y  # total number of frequency aware repeats
    low_freqs = 16  # B**2 - m
    block_sz = 4  # B
    
    config.eval_dir = f"output/evaluation/FA_EC_l{num_fa_length}r{num_fa_repeats}_{block_sz}by{block_sz}_low{low_freqs}"
    config.eval = d(
        eval_start=400000,
        n_samples=50000,
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
            "num_experts": 8,
            "router": "topk",
            "top_k": 6,
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
        Y_bound=[502.5],  # eta
        Y_mean=[-270.982, -0.725, 0.44, 0.107, 0.009, 0.154, -0.058, 0.0, -0.008, 0.019, 0.009, -0.012, 0.005, 0.002, 0.004, -0.002],  # eta
        Y_std=[182.433, 32.822, 34.997, 14.001, 17.621, 13.189, 5.791, 9.609, 9.805, 5.653, 4.634, 6.503, 4.964, 3.796, 3.44, 2.178],
        Y_min=[-502.5, -119.312, -122.375, -51.25, -58.906, -49.5, -20.703, -32.906, -33.094, -19.891, -15.891, -21.75, -16.969, -12.773, -11.648, -7.492],
        Y_max=[262.0, 112.375, 125.812, 49.0, 58.5, 46.5, 20.344, 32.906, 33.406, 19.953, 15.953, 22.0, 16.984, 12.812, 11.68, 7.461],
        Y_entropy=[5.883, 3.501, 3.575, 2.418, 2.75, 2.341, 1.479, 2.017, 2.044, 1.453, 1.316, 1.586, 1.365, 1.174, 1.108, 1.0],
        SNR_scale=4.0,
        greyscale=True,  # use greyscale images
        reweight=True,  # use loss reweighting based on entropy
        temperature=0.0,  # temperature for loss reweighting
        reweight_dim=-1,  # dimension to apply loss reweighting (1: channel-wise, 2: token-wise, -1: element-wise)
        frequency_aware_tokens=True,  # use frequency aware tokens
        tokenwise_normalization="Y_bound",  # 'minmax' or 'Y_bound' normalization for each token
        num_fa_length=num_fa_length,  # number of frequency aware coefficients length
        num_fa_repeats=num_fa_repeats,  # total number of frequency aware repeats
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
