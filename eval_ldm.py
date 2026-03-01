from tools.fid_score import calculate_fid_given_paths
from tools.is_lpips import calculate_lpips_score
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
import sde
from datasets import get_dataset
import tempfile
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import builtins
import libs.autoencoder
from libs.uvit import unpatchify
import os
import glob


def evaluate_checkpoints(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    results_dir = os.path.join(config.workdir, 'eval_results')
    if accelerator.is_main_process:
        os.makedirs(results_dir, exist_ok=True)
        utils.set_logger(log_level='info', fname=os.path.join(results_dir, 'eval.log'))
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)

    nnet = utils.get_nnet(**config.nnet).to(device)
    # nnet = accelerator.prepare(nnet)
    nnet_ema = utils.get_nnet(**config.nnet).to(device)
    nnet_ema.eval()

    # Create score_model with the loaded checkpoint
    # if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:
    #     def cfg_nnet(x, timesteps, y):
    #         _cond = nnet(x, timesteps, y=y)
    #         _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
    #         return _cond + config.sample.scale * (_cond - _uncond)
    #     score_model = sde.ScoreModel(cfg_nnet, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))
    # else:
    score_model = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))

    autoencoder = libs.autoencoder.get_model_diffusers(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    # Get all checkpoints
    ckpt_root = config.ckpt_root
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_root, '*.ckpt')),
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    eval_start = config.eval.get('eval_start', 0)
    ckpts = [ckpt for ckpt in ckpts if int(os.path.basename(ckpt).split('.')[0]) >= eval_start]
    
    if accelerator.is_main_process:
        logging.info(f"Found {len(ckpts)} checkpoints to evaluate")

    patch_size = config.encode.get('patch_size', None)
    
    # Prepare results directory
    
    if accelerator.is_main_process:
        os.makedirs(results_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    results = []

    # Evaluate each checkpoint
    for ckpt_path in ckpts:
        step = int(os.path.basename(ckpt_path).split('.')[0])
        
        if accelerator.is_main_process:
            logging.info(f"\n{'='*50}")
            logging.info(f"Evaluating checkpoint: {step}")
            logging.info(f"{'='*50}")
        
        try:
            # Load checkpoint
            logging.info(f'Loading checkpoint from {ckpt_path}')
            train_state = utils.TrainState(
                optimizer=None, 
                lr_scheduler=None,
                step=step,
                nnet=nnet, 
                nnet_ema=nnet_ema)
            train_state.resume(ckpt_root, step=step)
            
            accelerator.wait_for_everyone()
            
            # Define sampling function
            def sample_fn(_n_samples):
                _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
                if config.train.mode == 'uncond':
                    kwargs = dict()
                elif config.train.mode == 'cond':
                    kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
                else:
                    raise NotImplementedError

                if config.sample.algorithm == 'euler_maruyama_sde':
                    _z = sde.euler_maruyama(sde.ReverseSDE(score_model), _z_init, config.eval.get('sample_steps', 100), verbose=False, **kwargs)
                elif config.sample.algorithm == 'euler_maruyama_ode':
                    _z = sde.euler_maruyama(sde.ODE(score_model), _z_init, config.eval.get('sample_steps', 100), verbose=False, **kwargs)
                elif config.sample.algorithm == 'dpm_solver':
                    noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
                    model_fn = model_wrapper(
                        score_model.noise_pred,
                        noise_schedule,
                        time_input_type='0',
                        model_kwargs=kwargs
                    )
                    dpm_solver = DPM_Solver(model_fn, noise_schedule)
                    _z = dpm_solver.sample(
                        _z_init,
                        steps=config.eval.get('sample_steps', 100),
                        eps=1e-4,
                        adaptive_step_size=False,
                        fast_version=True,
                    )
                else:
                    raise NotImplementedError
                
                # Decode samples
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        if patch_size is not None and patch_size > 1:
                            z = unpatchify(_z, channels=autoencoder.config.latent_channels)
                        else:
                            b, tokens, c = _z.shape
                            h = w = int(tokens ** 0.5)
                            z = _z.reshape(b, h, w, c).permute(0, 3, 1, 2)
                        
                        z = z / autoencoder.config.scaling_factor
                        decoded = autoencoder.decode(z).sample
                return decoded
            
            logging.info(f'Sampling: n_samples={config.sample.n_samples}, mode={config.train.mode}')
            
            with tempfile.TemporaryDirectory() as temp_path:
                path = temp_path
                if accelerator.is_main_process:
                    os.makedirs(path, exist_ok=True)
                
                # Get LPIPS sample count (default: 2000)
                lpips_n_samples = config.eval.get('lpips_n_samples', 2000)
                fid_n_samples = config.sample.n_samples
                
                lpips_score = 0.0  # Initialize LPIPS score
                
                # Phase 1: Generate samples for LPIPS (smaller batch)
                logging.info(f'Phase 1: Generating {lpips_n_samples} samples for LPIPS...')
                utils.sample2dir(accelerator, path, lpips_n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
                
                # Calculate LPIPS early on smaller batch
                if accelerator.is_main_process:
                    logging.info("Computing LPIPS on initial batch...")
                    real_dir = config.eval.get('real_dir', 'data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs')
                    lpips_score = calculate_lpips_score(
                        path,
                        real_dir,
                        device,
                        batch_size=config.sample.get('lpips_batch_size', 32)
                    )
                    logging.info(f'LPIPS: {lpips_score:.4f}')
                
                # Phase 2: Generate remaining samples for FID
                if fid_n_samples > lpips_n_samples:
                    remaining_samples = fid_n_samples - lpips_n_samples
                    logging.info(f'Phase 2: Generating {remaining_samples} more samples for FID (total: {fid_n_samples})...')
                    utils.sample2dir(accelerator, path, remaining_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
                
                # Calculate FID on all samples
                if accelerator.is_main_process:
                    logging.info("Computing FID on all samples...")
                    fid_score = calculate_fid_given_paths((dataset.fid_stat, path))
                    logging.info(f'FID: {fid_score:.4f}')
                    
                    result = {
                        'step': step,
                        'fid': fid_score,
                        'lpips': lpips_score
                    }
                    results.append(result)
                    
                    # Save individual result
                    result_file = os.path.join(results_dir, f'metrics_step_{step}.txt')
                    with open(result_file, 'w') as f:
                        f.write(f"Step: {step}\n")
                        f.write(f"FID: {fid_score:.6f}\n")
                        f.write(f"LPIPS: {lpips_score:.6f}\n")
                
                accelerator.wait_for_everyone()
        
        except Exception as e:
            if accelerator.is_main_process:
                logging.error(f"Error evaluating checkpoint {step}: {str(e)}")
            accelerator.wait_for_everyone()
            continue
    
    # Save all results
    if accelerator.is_main_process:
        results_file = os.path.join(results_dir, 'all_metrics.txt')
        with open(results_file, 'w') as f:
            f.write("Step\tFID\tLPIPS\n")
            for result in results:
                f.write(f"{result['step']}\t{result['fid']:.6f}\t{result['lpips']:.6f}\n")
        
        logging.info(f"\nAll results saved to {results_file}")
        
        # Print summary
        logging.info("\n" + "="*50)
        logging.info("Evaluation Summary")
        logging.info("="*50)
        for result in results:
            logging.info(f"Step {result['step']}: FID={result['fid']:.4f}, LPIPS={result['lpips']:.4f}")
        
        if results:
            best_fid_result = min(results, key=lambda x: x['fid'])
            logging.info(f"\nBest FID: Step {best_fid_result['step']} with FID={best_fid_result['fid']:.4f}")


from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work directory.")


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir or config.get('workdir', 'exp_train')
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    evaluate_checkpoints(config)


if __name__ == "__main__":
    app.run(main)
