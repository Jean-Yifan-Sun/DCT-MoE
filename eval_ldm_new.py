import os
import glob
import tempfile
import torch
from torch import multiprocessing as mp
import accelerate
import ml_collections
from absl import logging, app, flags
from ml_collections import config_flags
import builtins
import utils
import sde
from datasets import get_dataset
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import libs.autoencoder
from libs.uvit import unpatchify
from tools.fid_score import calculate_fid_given_paths
from tools.is_lpips import calculate_lpips_score

def evaluate_checkpoints(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn', force=True)
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

    # 加载数据集（仅用于获取fid_stat和预处理函数）
    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)

    # 初始化模型
    nnet = utils.get_nnet(**config.nnet).to(device)
    nnet_ema = utils.get_nnet(**config.nnet).to(device)
    nnet_ema.eval()   # 确保EMA模型处于eval模式

    autoencoder = libs.autoencoder.get_model_diffusers(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    # 获取所有checkpoint
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

    # 构建score_model_ema（与训练eval_step一致）
    score_model_ema = sde.ScoreModel(
        nnet_ema,
        pred=config.pred,
        sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale)
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        if config.train.mode == 'uncond':
            kwargs = {}
        elif config.train.mode == 'cond':
            kwargs = {'y': dataset.sample_label(_n_samples, device=device)}
        else:
            raise NotImplementedError

        # 使用DPM-Solver，步数固定为50（与训练时eval_step一致）
        noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
        model_fn = model_wrapper(
            score_model_ema.noise_pred,
            noise_schedule,
            time_input_type='0',
            model_kwargs=kwargs
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        _z = dpm_solver.sample(
            _z_init,
            steps=50,
            eps=1e-4,
            adaptive_step_size=False,
            fast_version=True,
        )

        # 解码
        with torch.amp.autocast('cuda'), torch.no_grad():
            if patch_size is not None and patch_size > 1:
                z = unpatchify(_z, channels=autoencoder.config.latent_channels)
            else:
                b, tokens, c = _z.shape
                h = w = int(tokens ** 0.5)
                z = _z.reshape(b, h, w, c).permute(0, 3, 1, 2)
            z = z / autoencoder.config.scaling_factor
            decoded = autoencoder.decode(z).sample
        return decoded

    results = []
    for ckpt_path in ckpts:
        step = int(os.path.basename(ckpt_path).split('.')[0])
        if accelerator.is_main_process:
            logging.info(f"\n{'='*50}\nEvaluating checkpoint {step}\n{'='*50}")

        # 加载checkpoint到nnet_ema（训练时保存的是train_state，包含ema权重）
        train_state = utils.TrainState(
            optimizer=None,
            lr_scheduler=None,
            step=step,
            nnet=nnet,          # 仅用于占位，实际不会使用nnet
            nnet_ema=nnet_ema
        )
        train_state.resume(ckpt_root, step=step)   # 假设resume正确加载ema

        # 再次确保ema模型处于eval模式（resume可能改变模式）
        nnet_ema.eval()

        with tempfile.TemporaryDirectory() as temp_path:
            path = temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)

            # 阶段1：生成LPIPS所需样本
            lpips_n_samples = config.eval.get('lpips_n_samples', 2000)
            logging.info(f'Generating {lpips_n_samples} samples for LPIPS...')
            utils.sample2dir(accelerator, path, lpips_n_samples,
                             config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            # 计算LPIPS
            lpips_score = None
            if accelerator.is_main_process:
                real_dir = config.eval.get('real_dir', None)
                if real_dir and os.path.exists(real_dir):
                    lpips_score = calculate_lpips_score(
                        path,
                        real_dir,
                        device,
                        batch_size=config.sample.get('lpips_batch_size', 32)
                    )
                    logging.info(f'LPIPS@{step}: {lpips_score:.4f}')
                else:
                    logging.warning('real_dir not provided or invalid, skip LPIPS')

            # 阶段2：生成剩余样本以达到FID总数
            fid_n_samples = config.sample.n_samples
            if fid_n_samples > lpips_n_samples:
                remaining = fid_n_samples - lpips_n_samples
                logging.info(f'Generating {remaining} additional samples for FID...')
                utils.sample2dir(accelerator, path, remaining,
                                 config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            # 计算FID
            if accelerator.is_main_process:
                fid_score = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'FID@{step}: {fid_score:.4f}')

                result = {'step': step, 'fid': fid_score}
                if lpips_score is not None:
                    result['lpips'] = lpips_score
                results.append(result)

                # 保存单个结果
                result_file = os.path.join(results_dir, f'metrics_step_{step}.txt')
                with open(result_file, 'w') as f:
                    f.write(f"Step: {step}\n")
                    f.write(f"FID: {fid_score:.6f}\n")
                    if lpips_score is not None:
                        f.write(f"LPIPS: {lpips_score:.6f}\n")

            accelerator.wait_for_everyone()

    # 汇总所有结果
    if accelerator.is_main_process:
        results_file = os.path.join(results_dir, 'all_metrics.txt')
        with open(results_file, 'w') as f:
            f.write("Step\tFID\tLPIPS\n")
            for r in results:
                lp = f"{r.get('lpips', 0):.6f}" if 'lpips' in r else 'N/A'
                f.write(f"{r['step']}\t{r['fid']:.6f}\t{lp}\n")
        logging.info(f"\nAll results saved to {results_file}")

        # 打印最佳结果
        if results:
            best_fid = min(results, key=lambda x: x['fid'])
            logging.info(f"Best FID: step {best_fid['step']} with FID={best_fid['fid']:.4f}")

def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir or config.get('workdir', 'exp_train')
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    evaluate_checkpoints(config)

if __name__ == "__main__":
    app.run(main)