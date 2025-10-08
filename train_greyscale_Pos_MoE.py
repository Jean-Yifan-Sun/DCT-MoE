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

    # [DEBUG] Enable anomaly detection to find the operation that causes NaN gradients
    # torch.autograd.set_detect_anomaly(True)

    mp.set_start_method('spawn')
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour
    accelerator = accelerate.Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    # save config file
    if accelerator.is_main_process:
        config.to_yaml(os.path.join(config.workdir, 'config.yaml'))

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes
    logging.info(f'use {accelerator.num_processes} GPUs with batch size {mini_batch_size}/GPU')

    # log setting
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        wandb.init(
            project="dct-diffusion-positional",
            config=config.to_dict(),
            name=config.get('name','Undefined'),
            dir=config.ckpt_root
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
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
    tokenwise_normalization = config.dataset.get("tokenwise_normalization","z-score")
    logging.info(f'Using {tokenwise_normalization} for positional token normalization')

    # DP training is not yet adapted for this script, keeping the placeholder
    if config.private.use_dp:
        logging.warning("DP training with Opacus is not fully adapted for this script yet.")
        # keep track of training states (lr, opt, model)
        train_state = utils.initialize_train_state(config, device, use_opacus=False)
    else:
        # keep track of training states (lr, opt, model)
        train_state = utils.initialize_train_state(config, device, use_opacus=False)

    # wrap data_loader and model with accelerator for distributed training
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # variables for loss reweighting (adapted for positional tokens)
    if config.dataset.reweight:
        low2high_order = zigzag_order(config.dataset.block_sz)
        Y_entropy = np.array(config.dataset.Y_entropy)
        logging.info(f'using {Y_entropy} for loss reweighting')
        reweight = Y_entropy[low2high_order][:config.dataset.low_freqs]
        reweight = reweight / (reweight.sum() / reweight.shape[0])  # normalization
        reweight = torch.from_numpy(reweight).to(device=device).float()
        # Reshape for broadcasting: (1, low_freqs, 1)
        reweight = reweight.view(1, -1, 1)
        temperature = config.dataset.get("temperature", 1.0)
        logging.info(f'Using temperature {temperature} for loss reweighting')
    else:
        reweight = None
        temperature = None
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
        start_time = time.time()
         # 在反向传播后添加梯度裁剪
        try:
            optimizer.zero_grad()

            # 前向传播
            with accelerator.autocast():
                if config.train.mode == 'uncond':
                    main_loss, aux_loss = sde.LSimple(
                        score_model, _batch,
                        pred=config.pred,
                        use_moe=config.nnet.use_moe,
                        reweight=reweight, temperature=temperature
                    )
                elif config.train.mode == 'cond':
                    main_loss, aux_loss = sde.LSimple(
                        score_model,  _batch['image'], y=_batch['label'],
                        pred=config.pred,
                        use_moe=config.nnet.use_moe,
                        reweight=reweight, temperature=temperature
                    )
                try:    
                    alpha = config.nnet.MoE.get("aux_loss_alpha", 0.0)
                except:
                    alpha = config.nnet.MoH.get("aux_loss_alpha", 0.0)
                    
                total_loss = main_loss + alpha * aux_loss if alpha != 0.0 else main_loss

            # 检查损失是否有效
            # if not torch.all(torch.isfinite(total_loss)):
            #     logging.error(f"[Step {train_state.step}] Loss contains NaN or Inf! Total loss: {total_loss.item()}")
            #     raise RuntimeError("Loss contains NaN or Inf")

            # 反向传播
            accelerator.backward(total_loss)

            # 检查梯度是否有效
            # nan_params = []
            # for name, param in nnet.named_parameters():
            #     if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
            #         nan_params.append(name)
            #         logging.error(f"[Step {train_state.step}] Gradient of {name} contains NaN or Inf")
            #         param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1e6, neginf=-1e6)  # 修复无效梯度

            # if nan_params:
            #     raise RuntimeError(f"Non-finite gradients detected in parameters: {', '.join(nan_params)}")

            # 优化器更新
            optimizer.step()
            lr_scheduler.step()
            train_state.ema_update(config.get('ema_rate', 0.9999))

            _metrics = dict()
            _metrics['total_loss'] = accelerator.gather(total_loss.detach()).mean().item()
            _metrics['main_loss'] = accelerator.gather(main_loss.detach()).mean().item()
            if alpha != 0.0:
                _metrics['aux_loss'] = accelerator.gather(aux_loss.detach()).mean().item()
            
            train_state.step += 1
            _metrics['time_per_step'] = time.time() - start_time

            return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

        except Exception as e:
            logging.error(f"[Step {train_state.step}] Error occurred: {str(e)}")
            raise e


    def eval_step(n_samples, sample_steps, algorithm, path):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}, samples save into {path}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            kwargs = dict(use_moe=config.nnet.use_moe) if config.train.mode == 'uncond' else dict(y=dataset.sample_label(_n_samples, device=device), use_moe=config.nnet.use_moe)

            if algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
                model_fn = model_wrapper(score_model_ema.noise_pred, noise_schedule, time_input_type='0', model_kwargs=kwargs)
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                return dpm_solver.sample(_x_init, steps=sample_steps, eps=1e-4, adaptive_step_size=False, fast_version=True)
            elif algorithm == 'euler_maruyama_ode':
                return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
            else:
                raise NotImplementedError

        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)

        # Use the new positional token sampling function
        if tokenwise_normalization == 'z-score':
            utils.PositionalTokenSample2dir(
                accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
                img_sz=config.dataset.resolution, low_freqs=config.dataset.low_freqs,
                block_sz=config.dataset.block_sz, mean=config.dataset.Y_mean, std=config.dataset.Y_std
            )
        elif tokenwise_normalization == 'minmax':
            utils.PositionalTokenSample2dir(
                accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
                img_sz=config.dataset.resolution, low_freqs=config.dataset.low_freqs,
                block_sz=config.dataset.block_sz, min=config.dataset.Y_min, max=config.dataset.Y_max
            )
    

        _fid = 0
        if accelerator.is_main_process:
            _fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
            with open(os.path.join(config.workdir, f'eval_{algorithm}_{n_samples}.log'), 'a') as f:
                print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
            shutil.rmtree(path)

        _fid = torch.tensor(_fid, device=device)
        return accelerator.reduce(_fid, reduction='sum').item()


    """Training and Evaluation"""
    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        with torch.no_grad():
            if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
                logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                wandb.log(metrics, step=train_state.step)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
                grid_img_path = os.path.join(config.sample_dir, f'{train_state.step}.png')
                logging.info(f'Save a grid of 16 samples into {grid_img_path} by DPM-Solver')
                x_init = torch.randn(16, *dataset.data_shape, device=device)
                
                kwargs = dict(use_moe=config.nnet.use_moe)
                _y_init = None
                if config.train.mode == 'cond':
                    _y_init = dataset.sample_label(16, device=device)
                    kwargs['y'] = _y_init

                noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
                model_fn = model_wrapper(score_model_ema.noise_pred, noise_schedule, time_input_type='0', model_kwargs=kwargs)
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                samples = dpm_solver.sample(x_init, steps=config.sample.sample_steps, eps=1e-4, adaptive_step_size=False, fast_version=True)
                
                # Use the new positional token grid image function
                if tokenwise_normalization == 'z-score':
                    utils.PositionalTokenSamples_to_grid_image(
                        samples, labels=_y_init,
                        img_sz=config.dataset.resolution, low_freqs=config.dataset.low_freqs,
                        block_sz=config.dataset.block_sz, mean=config.dataset.Y_mean, std=config.dataset.Y_std,
                        grid_sz=4, path=grid_img_path
                    )
                elif tokenwise_normalization == 'minmax':
                    utils.PositionalTokenSamples_to_grid_image(
                        samples, labels=_y_init,
                        img_sz=config.dataset.resolution, low_freqs=config.dataset.low_freqs,
                        block_sz=config.dataset.block_sz, min=config.dataset.Y_min, max=config.dataset.Y_max,
                        grid_sz=4, path=grid_img_path
                    )

                wandb.log({"samples": wandb.Image(grid_img_path)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        save_start = config.sample.get('save_start', 10000)
        if train_state.step >= save_start and train_state.step % config.train.save_interval == 0:
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            # calculate fid of the saved checkpoint using DPM-Solver (NFE=50)
            fid_dpm = eval_step(n_samples=config.sample.n_samples, sample_steps=50,
                            algorithm='dpm_solver', path=os.path.join(config.sample_dir, f'{config.name}_{train_state.step}_dpm'))
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            # calculate fid of the saved checkpoint using Euler ODE Solver (NFE=100)
            fid_euler = eval_step(n_samples=config.sample.n_samples, sample_steps=100,
                            algorithm='euler_maruyama_ode', 
                            path=os.path.join(config.sample_dir, f'{config.name}_{train_state.step}_euler'))
            torch.cuda.empty_cache()
            
            if accelerator.is_main_process:
                wandb.log({f"fid{config.sample.n_samples}_dpm_solver": fid_dpm}, 
                          step=train_state.step)
                wandb.log({f"fid{config.sample.n_samples}_euler_maruyama_ode": fid_euler},
                           step=train_state.step)
            accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
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
    config.workdir = FLAGS.workdir or 'exp_train_pos_token'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)