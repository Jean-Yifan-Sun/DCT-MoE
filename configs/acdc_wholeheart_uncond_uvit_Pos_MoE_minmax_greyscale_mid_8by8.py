import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.name = 'acdc_wholeheart_uncond_uvit_Pos_MoE_minmax_greyscale_mid_8by8'

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
        tokens=64,  # number of tokens to the network
        in_chans=144, # number of input channels (for greyscale 96x96 images with 8x8 blocks, there are 144 DCT channels)
        low_freqs=64,  # B**2 - m
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=True,
        num_classes=-1,
        use_moe=True,
        MoE={
            "depth": 1,
            "num_experts": 4,
            "router": "topk",
            "top_k": 2,
            "noise_eps": 1e-2,
            "aux_loss_alpha": 0.001
        },

    )

    config.dataset = d(
        name='acdc_uncond',
        path='data/scratch/datasets/ACDC/Unlabeled/Wholeheart',
        dataset_type='wholeheart',  # use wholeheart dataset
        resolution=96,
        tokens=64,  # number of tokens to the network
        low_freqs=64,  # B**2 - m
        block_sz=8,  # B
        Y_mean=[483.731, -3.252, 1.744, 0.333, 0.046, 0.864, -0.305, -0.031, 0.056, 0.254, 0.05, 0.05, -0.113, -0.022, 0.113, -0.123, -0.005, -0.108, -0.033, 0.013, 0.049, 0.02, 0.007, -0.017, -0.02, -0.0, 0.023, 0.037, -0.009, -0.015, -0.007, 0.001, -0.011, 0.013, -0.013, 0.002, -0.016, 0.005, 0.022, -0.014, -0.007, -0.014, 0.002, 0.01, 0.003, 0.001, 0.015, -0.004, -0.003, 0.012, -0.011, -0.017, 0.001, 0.0, -0.0, 0.009, 0.009, 0.002, 0.002, 0.001, -0.003, -0.002, -0.003, 0.001],  # eta
        Y_std=[333.013, 85.303, 93.012, 44.02, 50.027, 40.348, 22.733, 30.904, 31.719, 24.094, 15.301, 20.141, 23.215, 19.631, 14.377, 9.175, 12.891, 16.177, 16.349, 13.388, 9.689, 6.159, 8.992, 11.597, 12.466, 11.243, 8.584, 6.31, 4.288, 6.008, 7.872, 9.26, 9.466, 8.138, 5.776, 3.418, 3.294, 5.392, 6.989, 7.389, 6.839, 5.695, 4.214, 4.041, 5.138, 5.718, 5.704, 4.78, 3.121, 2.806, 3.962, 4.548, 4.368, 3.686, 3.151, 3.433, 3.119, 2.338, 1.877, 2.358, 2.536, 1.865, 1.549, 1.312],
        Y_min=[23.875, -289.75, -306.75, -157.125, -155.625, -142.25, -78.938, -98.125, -99.938, -81.125, -52.875, -64.688, -74.438, -64.688, -50.375, -31.828, -42.656, -53.25, -52.75, -43.688, -32.656, -21.016, -29.469, -37.344, -40.219, -36.594, -28.484, -21.75, -14.898, -20.172, -25.938, -30.079, -30.484, -26.406, -19.25, -12.156, -11.805, -17.828, -22.406, -23.75, -22.312, -18.797, -14.461, -13.734, -16.781, -18.469, -18.266, -15.703, -11.062, -9.938, -12.969, -14.602, -14.219, -12.539, -10.992, -11.164, -10.297, -8.414, -7.078, -8.148, -9.203, -7.004, -6.078, -5.168],
        Y_max=[1474.0, 266.75, 314.25, 142.625, 151.75, 132.625, 77.688, 99.875, 101.625, 83.125, 50.5, 65.438, 74.438, 64.438, 47.875, 31.062, 42.188, 52.594, 52.688, 43.406, 32.75, 20.75, 29.5, 37.562, 40.312, 37.188, 28.688, 21.391, 14.922, 20.031, 26.141, 30.204, 30.453, 26.484, 19.266, 12.281, 11.602, 17.984, 22.547, 23.75, 22.328, 18.734, 14.461, 13.805, 16.844, 18.531, 18.266, 15.758, 11.102, 9.953, 12.984, 14.555, 14.195, 12.531, 11.016, 11.344, 10.312, 8.445, 7.055, 8.133, 9.211, 6.953, 5.949, 5.172],
        # Y_min=[0.0]*64,  # for min-max normalization
        # Y_max=[1000.0]*64,  # for min-max normalization
        Y_entropy=[5.303, 3.42, 3.521, 2.533, 2.747, 2.432, 1.78, 2.144, 2.177, 1.839, 1.407, 1.662, 1.817, 1.632, 1.363, 1.052, 1.278, 1.446, 1.454, 1.304, 1.082, 1.0, 1.0, 1.192, 1.248, 1.175, 1.0, 1.0, 1.0, 1.0, 1.0, 1.022, 1.033, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # for loss reweighting
        SNR_scale=4.0,
        greyscale=True,  # use greyscale images
        positional_tokens=True,  # use positional tokens
        tokenwise_normalization="minmax",  # use token-wise normalization
        reweight=True,  # use loss reweighting
        temperature=1.0,  # temperature for loss reweighting
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
