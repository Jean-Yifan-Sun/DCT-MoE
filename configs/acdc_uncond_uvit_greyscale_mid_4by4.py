import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=500000,
        batch_size=256,
        mode='uncond',
        log_interval=100,
        eval_interval=25000,
        save_interval=25000,
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
        tokens=144,  # number of tokens to the network
        low_freqs=16,  # B**2 - m
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='acdc_uncond',
        path='data/scratch/datasets/ACDC/JPGs',
        resolution=96,
        tokens=144,  # number of tokens to the network
        low_freqs=16,  # B**2 - m
        block_sz=4,  # B
        Y_bound=[502.0],  # eta
        Y_std=[5.854, 3.48, 2.358, 1.471, 3.493, 2.733, 2.039, 1.269, 2.369, 2.047, 1.485, 0.998, 1.381, 1.166, 0.998, 0.999],
        Cb_std=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        Cr_std=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        SNR_scale=4.0,
        greyscale=True,  # use greyscale images
    )

    config.sample = d(
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
