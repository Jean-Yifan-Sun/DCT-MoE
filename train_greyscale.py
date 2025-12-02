import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import numpy as np
import shutil
from DCT_utils import zigzag_order, reverse_zigzag_order
from opacus import PrivacyEngine
from torch.utils.data import Subset
import wandb,time


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour
    accelerator = accelerate.Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes  # batch per GPU
    logging.info(f'use {accelerator.num_processes} GPUs with batch size {mini_batch_size}/GPU')

    # log setting
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        wandb.init(
            project="dct-diffusion",  # 设置你的项目名
            config=config.to_dict(),  # 记录配置
            name=config.get('name','Undefined'),  # 使用工作目录名作为运行名称
            dir=config.ckpt_root  # wandb文件保存位置
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
        yaml = config.to_yaml()
        os.makedirs(config.workdir, exist_ok=True)
        with open(os.path.join(config.workdir, 'config.yaml'), 'w') as f:
            f.write(yaml)
            f.close()
        logging.info(f'Config file saved to {os.path.join(config.workdir, "config.yaml")}')

    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    # Dataset and DataLoader
    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=(config.train.mode == 'cond'))
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=2, pin_memory=True, persistent_workers=True)
    logging.info(f'dataset samples: {len(train_dataset)}')

    # Use Opacus for DP training
    if config.private.use_dp and config.private.dp_method == 'dpsgd':
        logging.info('Using Opacus for DP training')
        # Calculate global batch size and remainder
        global_batch_size = config.train.batch_size
        remainder = len(train_dataset) % global_batch_size

        # The number of samples to keep
        num_samples_to_keep = len(train_dataset) - remainder
        # Create a new dataset containing only the samples we want to keep
        indices = np.arange(num_samples_to_keep)
        truncated_dataset = Subset(train_dataset, indices)

        logging.info(f"Original dataset size: {len(train_dataset)}")
        logging.info(f"Truncated dataset size: {len(truncated_dataset)} (divisible by {global_batch_size})")

        train_dataset_loader = DataLoader(truncated_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=False,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
        opacus_params = dict(
            data_loader=train_dataset_loader,
            accoutant=config.private.accountant,
            secure_mode=config.private.secure_mode,
            noise_multiplier=config.private.noise_multiplier,
            epochs=config.train.n_steps // (len(train_dataset) // mini_batch_size),
            target_epsilon=config.private.target_epsilon,
            target_delta=config.private.target_delta,
            max_grad_norm=config.private.max_grad_norm,
            epsilon_first=True,  # use epsilon_first for Opacus
        )
        train_state, privacy_engine = utils.initialize_train_state(config, device, use_opacus=True, **opacus_params)
        train_dataset_loader = train_state.dataloader
        _noise_multiplier = train_state.optimizer.noise_multiplier  # update noise_multiplier from Opacus
        logging.info(f"PrivacyEngine added to the model. Setting epsilon={config.private.target_epsilon}. Using noise_multiplier={_noise_multiplier} and max_grad_norm={config.private.max_grad_norm}.")
        accelerator.even_batches = False
    else:
        # keep track of training states (lr, opt, model)
        train_state = utils.initialize_train_state(config, device, use_opacus=False)

    # wrap data_loader and model with accelerator for distributed training
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # variables for loss reweighting
    low2high_order = zigzag_order(config.dataset.block_sz)  # list
    reverse_order = reverse_zigzag_order(config.dataset.block_sz)  # list
    if config.dataset.reweight:
        Y_entropy = np.array(config.dataset.Y_entropy)
        logging.info(f'using {Y_entropy} for loss reweighting')
        reweight = Y_entropy[:config.dataset.low_freqs]
        # reweight = reweight / (reweight.sum() / reweight.shape[0])  # normalization
        reweight = torch.from_numpy(reweight).to(device=device).float()
        reweight = torch.cat((reweight, reweight, reweight, reweight))
        assert reweight.shape[0] == config.dataset.low_freqs * 4
        # reweight = reweight.view(1, -1, 1)
        temperature = config.dataset.get("temperature", 1.0)
        fa_transform_fn = train_dataset.FA_transform if hasattr(train_dataset, 'FA_transform') else None
        logging.info(f'Using temperature {temperature} for loss reweighting')
        reweight_dim = config.dataset.get("reweight_dim", 2)  # default token-wise reweighting
        criterion = sde.EntropyWeightedMSELoss(reweight, temperature=temperature, normalize_weights=True, dim=reweight_dim, fa_transform_fn=fa_transform_fn)
        logging.info(f'Using temperature {temperature} for loss reweighting')
    else:
        reweight = None
        temperature = None
        fa_transform_fn = None
        criterion=None
        logging.info('Not using loss reweighting.')


    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # wrap network with diffusion framework
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))

    def train_step(_batch):
        _metrics = dict()
        start_time = time.time()
        optimizer.zero_grad()

        """GFLOPs calculation (set batch_size = 1)"""
        # from thop import profile
        # t = torch.ones((_batch.shape[0])).to(_batch.device)
        # flops, params = profile(nnet, inputs=(_batch, t))
        # gflops = flops / 1e9
        # print(f"gFLOPs: {gflops}")
        # print(f"number of parameters: {params}")
        # raise ValueError

        if config.train.mode == 'uncond':
            loss = sde.LSimple(score_model, _batch,
                                pred=config.pred,
                                criterion=criterion)
             # mean over batch
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch['image'],
                                pred=config.pred, y=_batch['label'], 
                                criterion=criterion)
            
        else:
            raise NotImplementedError(config.train.mode)
        
        accelerator.backward(loss)
        
        _metrics['loss'] = accelerator.gather(loss.detach()).mean().item()       
        
        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        time_elapsed = time.time() - start_time
        _metrics['time_per_step'] = time_elapsed
        if config.private.use_dp and config.private.dp_method == 'dpsgd':
            # update privacy engine
            _metrics['epsilon'] = privacy_engine.get_epsilon(config.private.target_delta)
            # _metrics['noise_multiplier'] = optimizer.noise_multiplier
            # _metrics['max_grad_norm'] = optimizer.max_grad_norm

        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)


    def eval_step(n_samples, sample_steps, algorithm, Y_bound, path):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}, samples save into {path}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            kwargs = dict(use_moe=config.nnet.use_moe) if config.train.mode == 'uncond' else dict(y=dataset.sample_label(_n_samples, device=device), use_moe=config.nnet.use_moe)

            if algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_ode':
                return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
                model_fn = model_wrapper(
                    score_model_ema.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                return dpm_solver.sample(
                    _x_init,
                    steps=sample_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError

        # create an empty folder to save generated images
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)

        # generate samples
        utils.DCTsample2dir_greyscale(
            accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
            tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
            reverse_order=reverse_order, resolution=config.dataset.resolution,
            block_sz=config.dataset.block_sz, Y_bound=Y_bound
        )

        # FID computation
        _fid = 0
        if accelerator.is_main_process:
            _fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
            with open(os.path.join(config.workdir, f'eval_{algorithm}_{n_samples}.log'), 'a') as f:
                print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
            shutil.rmtree(path)  # remove all generated images

        _fid = torch.tensor(_fid, device=device)
        _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()


    """Training and Evaluation"""
    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        # logging
        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            # logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)
        accelerator.wait_for_everyone()

        # visualize generated images by DPM-Solver
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            grid_img_path = os.path.join(config.sample_dir, f'{train_state.step}.png')
            logging.info(f'Save a grid of 16 samples into {grid_img_path} by DPM-Solver')
            x_init = torch.randn(16, *dataset.data_shape, device=device)

            if config.train.mode == 'uncond':
                kwargs = dict(use_moe=config.nnet.use_moe)
            elif config.train.mode == 'cond':
                _y_init = dataset.sample_label(16, device=device)
                kwargs = dict(y=_y_init, use_moe=config.nnet.use_moe)
            else:
                raise NotImplementedError

            noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
            model_fn = model_wrapper(
                score_model_ema.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            samples = dpm_solver.sample(
                x_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
            if config.train.mode == 'uncond':
                utils.DCTsamples_to_grid_image_greyscale(
                    samples, tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
                    block_sz=config.dataset.block_sz, reverse_order=reverse_order,
                    resolution=config.dataset.resolution, grid_sz=4, path=grid_img_path, Y_bound=config.dataset.Y_bound
                )
            elif config.train.mode == 'cond':
                utils.DCTsamples_to_grid_image_greyscale(
                    samples, 
                    labels=_y_init,  # use the same labels as the sampled images
                    tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
                    block_sz=config.dataset.block_sz, reverse_order=reverse_order,
                    resolution=config.dataset.resolution, grid_sz=4, path=grid_img_path, Y_bound=config.dataset.Y_bound
                )

            wandb.log({
                    "samples": wandb.Image(grid_img_path),
                    "step": train_state.step
                })
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        # save ckpt and FID evaluation
        save_start = config.sample.get('save_start', 10000)
        if train_state.step >= save_start and train_state.step % config.train.save_interval == 0:
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

            # calculate fid of the saved checkpoint using DPM-Solver (NFE=50)
            fid_dpm = eval_step(n_samples=config.sample.n_samples, sample_steps=50,
                            algorithm='dpm_solver', Y_bound=config.dataset.Y_bound,
                            path=os.path.join(config.sample_dir, f'{config.name}_{train_state.step}_dpm'))
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

            # calculate fid of the saved checkpoint using Euler ODE Solver (NFE=100)
            # fid_euler = eval_step(n_samples=config.sample.n_samples, sample_steps=100,
            #                 algorithm='euler_maruyama_ode', Y_bound=config.dataset.Y_bound,
            #                 path=f'{config.sample.path}_eulerODE')
            # torch.cuda.empty_cache()
            
            if accelerator.is_main_process:
                wandb.log({
                    f"fid{config.sample.n_samples}_dpm_solver": fid_dpm,
                    "step": train_state.step
                })
                # wandb.log({
                #     f"fid{config.sample.n_samples}_euler_maruyama_ode": fid_euler,
                #     "step": train_state.step
                # })
            accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    del metrics
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()
    logging.info(f'all done!')


from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir or 'exp_train'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
