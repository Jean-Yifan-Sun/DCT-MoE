import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
import cv2
from PIL import Image
from DCT_utils import idct_transform, combine_blocks
import threading
import queue
from multiprocessing import Process, Queue, Event, cpu_count
import time


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    elif name == 'uvit_greyscale':
        from libs.uvit import UViT_greyscale
        return UViT_greyscale(**kwargs)
    elif name == 'uvit_greyscale_cond':
        from libs.uvit import UViT_greyscale_cond
        return UViT_greyscale_cond(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        if p_name not in param_dict_src:
            # 跳过 model_src 没有的参数
            logging.warning(f'Parameter {p_name} not found in source model, skipping EMA update.')
            continue
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None, dataloader=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.dataloader = dataloader

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            if self.dataloader is not None:
                ema(self.nnet_ema, self.nnet._module, rate)
            else:
                ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device, use_opacus=False, **opacus_params):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    if use_opacus:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine(
            accountant=opacus_params.get("accountant", "prv"),
            secure_mode=opacus_params.get("secure_mode", False),
        )
        epsilon_first = opacus_params.get("epsilon_first", False)
        if epsilon_first:
            nnet, optimizer, _data_loader = privacy_engine.make_private_with_epsilon(
                module=nnet,
                optimizer=optimizer,
                data_loader=opacus_params["data_loader"],
                epochs=opacus_params.get("epochs", 1),
                target_epsilon=opacus_params.get("target_epsilon", 8),
                target_delta=opacus_params.get("target_delta", 1e-5),
                max_grad_norm=opacus_params.get("max_grad_norm", 1.0),
            )
        else:
            nnet, optimizer, _data_loader = privacy_engine.make_private(
                module=nnet,
                optimizer=optimizer,
                data_loader=opacus_params["data_loader"],
                epochs=opacus_params.get("epochs", 1),
                noise_multiplier=opacus_params.get("noise_multiplier", 0.5),
                max_grad_norm=opacus_params.get("max_grad_norm", 1.0),
            )
        train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema, dataloader=_data_loader)
        train_state.ema_update(0)
        train_state.to(device)
        return train_state, privacy_engine

    else:
        train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                                nnet=nnet, nnet_ema=nnet_ema)
        train_state.ema_update(0)
        train_state.to(device)
        return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def DCT_to_RGB(sample, tokens=0, low_freqs=0, block_sz=0, reverse_order=None, resolution=0, Y_bound=None):
    # cb_index = [i for i in range(4, tokens, 6)]
    # cr_index = [i for i in range(5, tokens, 6)]
    # y_index = [i for i in range(0, tokens) if i not in cb_index and i not in cr_index]
    # assert len(y_index) + len(cb_index) + len(cr_index) == tokens
    # y_tokens = int((tokens / 6) * 4)
    # cb_tokens = int(tokens / 6)

    num_y_blocks = tokens * 4
    num_cb_blocks = tokens
    cb_blocks_per_row = int((resolution / block_sz) / 2)
    Y_blocks_per_row = int(resolution / block_sz)

    assert sample.shape == (tokens, low_freqs*6)
    sample = np.clip(sample, -2, 2)  # clamp into [-1, 1]
    sample = sample.reshape(tokens, 6, low_freqs)  # (tokens, 6, low_freqs)

    # fill up DCT coes
    DCT = np.zeros((tokens, 6, block_sz * block_sz))  # (tokens, 6, B**2)
    DCT[:, :, :low_freqs] = sample
    DCT = DCT[..., reverse_order]  # convert the low to high freq order back to 8*8 order

    Y_bound = np.array(Y_bound)
    DCT_Y = DCT[:, :4, :] * Y_bound  # (tokens, 4, B**2)
    DCT_Cb = DCT[:, 4, :] * Y_bound  # (tokens, B**2)
    DCT_Cr = DCT[:, 5, :] * Y_bound  # (tokens, B**2)

    DCT_Cb = DCT_Cb.reshape(num_cb_blocks, block_sz, block_sz)  # (tokens, B, B)
    DCT_Cr = DCT_Cr.reshape(num_cb_blocks, block_sz, block_sz)  # (tokens, B, B)

    y_blocks = []
    for row in range(cb_blocks_per_row):  # 16 cb/cr blocks, so 4*4 spatial blocks
        tem_ls = []
        for col in range(cb_blocks_per_row):
            ind = row * cb_blocks_per_row + col
            y_blocks.append(DCT_Y[ind, 0, :])
            y_blocks.append(DCT_Y[ind, 1, :])
            tem_ls.append(DCT_Y[ind, 2, :])
            tem_ls.append(DCT_Y[ind, 3, :])
        for ele in tem_ls:
            y_blocks.append(ele)
    DCT_Y = np.array(y_blocks).reshape(num_y_blocks, block_sz, block_sz)  # (Y_blocks, B, B)

    # Apply Inverse DCT on each block
    idct_y_blocks = idct_transform(DCT_Y)
    idct_cb_blocks = idct_transform(DCT_Cb)
    idct_cr_blocks = idct_transform(DCT_Cr)

    # Combine blocks back into images
    height, width = resolution, resolution
    y_reconstructed = combine_blocks(idct_y_blocks, height, width, block_sz)
    cb_reconstructed = combine_blocks(idct_cb_blocks, int(height / 2), int(width / 2), block_sz)
    cr_reconstructed = combine_blocks(idct_cr_blocks, int(height / 2), int(width / 2), block_sz)

    # Upsample Cb and Cr to original size
    cb_upsampled = cv2.resize(cb_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
    cr_upsampled = cv2.resize(cr_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)

    # Step 5: Convert YCbCr back to RGB
    R = y_reconstructed + 1.402 * (cr_upsampled - 128)
    G = y_reconstructed - 0.344136 * (cb_upsampled - 128) - 0.714136 * (cr_upsampled - 128)
    B = y_reconstructed + 1.772 * (cb_upsampled - 128)

    rgb_reconstructed = np.zeros((height, width, 3))
    rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
    rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
    rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)

    # Convert to uint8
    rgb_reconstructed = np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels

    return rgb_reconstructed

def DCT_to_greyscale(sample, tokens=0, low_freqs=0, block_sz=0, reverse_order=None, resolution=0, Y_bound=None):
    """
    将 DCT token 还原为灰度图像。
    sample: (tokens, low_freqs*4)
    """
    num_y_blocks = tokens * 4
    cb_blocks_per_row = int((resolution / block_sz) / 2)
    Y_blocks_per_row = int(resolution / block_sz)

    assert sample.shape == (tokens, low_freqs*4)
    sample = np.clip(sample, -2, 2)
    sample = sample.reshape(tokens, 4, low_freqs)  # (tokens, 4, low_freqs)

    # 填充 DCT 系数
    DCT_Y = np.zeros((tokens, 4, block_sz * block_sz))
    DCT_Y[:, :, :low_freqs] = sample
    DCT_Y = DCT_Y[..., reverse_order]  # 恢复原始顺序

    Y_bound = np.array(Y_bound)
    DCT_Y = DCT_Y * Y_bound  # 反归一化

    # 还原 Y blocks 顺序
    y_blocks = []
    for row in range(cb_blocks_per_row):  # 16 cb/cr blocks, so 4*4 spatial blocks
        tem_ls = []
        for col in range(cb_blocks_per_row):
            ind = row * cb_blocks_per_row + col
            y_blocks.append(DCT_Y[ind, 0, :])
            y_blocks.append(DCT_Y[ind, 1, :])
            tem_ls.append(DCT_Y[ind, 2, :])
            tem_ls.append(DCT_Y[ind, 3, :])
        for ele in tem_ls:
            y_blocks.append(ele)
    DCT_Y = np.array(y_blocks).reshape(num_y_blocks, block_sz, block_sz)  # (Y_blocks, B, B)

    # 逆 DCT
    idct_y_blocks = idct_transform(DCT_Y)
    y_reconstructed = combine_blocks(idct_y_blocks, resolution, resolution, block_sz)

    # 转为 uint8 灰度图
    grey_img = np.clip(y_reconstructed, 0, 255).astype(np.uint8)
    return grey_img

def DCTsamples_to_grid_image(samples, tokens=0, low_freqs=0, block_sz=0,
                             reverse_order=None, resolution=0, grid_sz=0, path=None, Y_bound=None):
    samples = samples.detach().cpu().numpy()
    rgb_imgs = []
    for sample in samples:
        rgb_reconstructed = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
        rgb_imgs.append(rgb_reconstructed)
    rgb_imgs = np.array(rgb_imgs)
    img_sz = rgb_imgs.shape[1]

    # Fill the grid image with the 36 smaller images
    grid_image = np.zeros((grid_sz * img_sz, grid_sz * img_sz, 3), dtype=np.uint8)
    for i in range(grid_sz):
        for j in range(grid_sz):
            idx = i * grid_sz + j
            if idx < rgb_imgs.shape[0]:
                grid_image[i * img_sz:(i + 1) * img_sz, j * img_sz:(j + 1) * img_sz, :] = rgb_imgs[idx]

    # Convert the NumPy array to an image and save or show it
    final_image = Image.fromarray(grid_image)
    final_image.save(path)

def DCTsamples_to_grid_image_greyscale(samples, labels=None, tokens=0, low_freqs=0, block_sz=0,
                                       reverse_order=None, resolution=0, grid_sz=0, path=None, Y_bound=None):
    samples = samples.detach().cpu().numpy()
    grey_imgs = []
    for sample in samples:
        grey_img = DCT_to_greyscale(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
        grey_imgs.append(grey_img)
    grey_imgs = np.array(grey_imgs)
    img_sz = grey_imgs.shape[1]

    if labels is not None:
        labels = labels.detach().cpu().numpy()
        label_list = []
        for i in range(labels.shape[0]):
            label_list.append(labels[i,:, :])
        grey_labs = []
        for label in label_list:
            grey_lab = DCT_to_greyscale(label, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
            grey_labs.append(grey_lab)
        grey_labs = np.array(grey_labs)
        assert img_sz == grey_labs.shape[1], "Image size and label size must match."
        grid_image = np.zeros((grid_sz * img_sz * 2, grid_sz * img_sz), dtype=np.uint8)
        for i in range(grid_sz):
            for j in range(grid_sz):
                idx = i * grid_sz + j
                if idx < grey_imgs.shape[0]:
                    grid_image[2 * i * img_sz:(2 * i + 1) * img_sz, j * img_sz:(j + 1) * img_sz] = grey_imgs[idx]
                    grid_image[(2 * i + 1) * img_sz:(2 * i + 2) * img_sz, (j + 0) * img_sz:(j + 1) * img_sz] = grey_labs[idx]
    else:
        # Fill the grid image with the grid_sz*grid_sz smaller images
        grid_image = np.zeros((grid_sz * img_sz, grid_sz * img_sz), dtype=np.uint8)
        for i in range(grid_sz):
            for j in range(grid_sz):
                idx = i * grid_sz + j
                if idx < grey_imgs.shape[0]:
                    grid_image[i * img_sz:(i + 1) * img_sz, j * img_sz:(j + 1) * img_sz] = grey_imgs[idx]

    # Convert the NumPy array to a greyscale image and save
    final_image = Image.fromarray(grid_image, mode='L')
    final_image.save(path)

def DCTsample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn,
                  tokens=0, low_freqs=0, reverse_order=None, resolution=0, block_sz=8, Y_bound=None):
    os.makedirs(path, exist_ok=True)
    batch_size = mini_batch_size * accelerator.num_processes
    num_iterations = n_samples // batch_size + 1
    print(f'using eta {Y_bound} for sampling')
    world_size = accelerator.state.num_processes
    local_rank = accelerator.state.local_process_index

    for i in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = sample_fn(mini_batch_size)
        samples = samples.detach().cpu().numpy()

        # distributed save
        for b_id in range(mini_batch_size):
            img_id = i * mini_batch_size * world_size + local_rank * mini_batch_size + b_id
            rgb_reconstructed = DCT_to_RGB(samples[b_id], tokens, low_freqs, block_sz, reverse_order, resolution,
                                           Y_bound)

            if img_id >= n_samples:
                break
            cv2.imwrite(os.path.join(path, f"{img_id}.jpg"), cv2.cvtColor(rgb_reconstructed, cv2.COLOR_RGB2BGR))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'generated {len(os.listdir(path))} images...')
        assert len(os.listdir(path)) == n_samples

def DCTsample2dir_greyscale(accelerator, path, n_samples, mini_batch_size, sample_fn,
                  tokens=0, low_freqs=0, reverse_order=None, resolution=0, block_sz=8, Y_bound=None):
    os.makedirs(path, exist_ok=True)
    batch_size = mini_batch_size * accelerator.num_processes
    num_iterations = n_samples // batch_size + 1
    print(f'using eta {Y_bound} for sampling')
    world_size = accelerator.state.num_processes
    local_rank = accelerator.state.local_process_index

    for i in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = sample_fn(mini_batch_size)
        samples = samples.detach().cpu().numpy()

        # distributed save
        for b_id in range(mini_batch_size):
            img_id = i * mini_batch_size * world_size + local_rank * mini_batch_size + b_id
            grey_img = DCT_to_greyscale(samples[b_id], tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)

            if img_id >= n_samples:
                break
            cv2.imwrite(os.path.join(path, f"{img_id}.jpg"), grey_img)  # 单通道灰度图

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'generated {len(os.listdir(path))} images...')
        assert len(os.listdir(path)) == n_samples

def worker_thread(sample_queue, stop_event, tokens, low_freqs, reverse_order, resolution, block_sz, Y_bound, path):
    # Background worker function: convert DCT to RGB and save images
    while not stop_event.is_set() or not sample_queue.empty():
        try:
            img_id, sample = sample_queue.get(timeout=1)
            rgb = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
            cv2.imwrite(os.path.join(path, f"{img_id}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            sample_queue.task_done()
        except queue.Empty:
            continue


def DCTsample2dir_threading(accelerator, path, n_samples, mini_batch_size, sample_fn,
                  tokens=0, low_freqs=0, reverse_order=None, resolution=0, block_sz=8, Y_bound=None):
    # DCT generation keeps on running; DCT_to_RGB will be run on the background

    os.makedirs(path, exist_ok=True)
    batch_size = mini_batch_size * accelerator.num_processes
    num_iterations = n_samples // batch_size + 1
    print(f'using eta {Y_bound} for sampling')
    world_size = accelerator.state.num_processes
    local_rank = accelerator.state.local_process_index

    # Threading setup
    sample_queue = queue.Queue()  # set maxsize=1024 if your RAM is limited
    stop_event = threading.Event()
    worker = threading.Thread(
        target=worker_thread,
        args=(sample_queue, stop_event, tokens, low_freqs, reverse_order, resolution, block_sz, Y_bound, path),
    )
    worker.start()

    # Main loop: generate and enqueue samples
    for i in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = sample_fn(mini_batch_size)
        samples = samples.detach().cpu().numpy()

        for b_id in range(mini_batch_size):
            img_id = i * mini_batch_size * world_size + local_rank * mini_batch_size + b_id
            if img_id >= n_samples:
                break
            sample_queue.put((img_id, samples[b_id]))

    # Clean shutdown
    sample_queue.join()
    stop_event.set()
    worker.join()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'generated {len(os.listdir(path))} images...')
        assert len(os.listdir(path)) == n_samples


def worker_process(sample_queue, stop_event, tokens, low_freqs, reverse_order, resolution, block_sz, Y_bound, path, worker_id):
    while not stop_event.is_set() or not sample_queue.empty():
        try:
            img_id, sample = sample_queue.get(timeout=1)
            rgb = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
            cv2.imwrite(os.path.join(path, f"{img_id}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            continue


def DCTsample2dir_multiprocess(accelerator, path, n_samples, mini_batch_size, sample_fn,
                  tokens=0, low_freqs=0, reverse_order=None, resolution=0, block_sz=8, Y_bound=None,
                  num_workers=None):
    os.makedirs(path, exist_ok=True)
    batch_size = mini_batch_size * accelerator.num_processes
    num_iterations = n_samples // batch_size + 1
    print(f'Using eta {Y_bound} for sampling')
    world_size = accelerator.state.num_processes
    local_rank = accelerator.state.local_process_index

    # Multiprocessing setup
    if num_workers is None:
        num_workers = min(os.cpu_count(), 16)
    sample_queue = Queue()  # set maxsize=1024 if your RAM is limited
    stop_event = Event()

    workers = []
    print(f"using {num_workers} num_workers")
    for n in range(num_workers):
        p = Process(
            target=worker_process,
            args=(sample_queue, stop_event, tokens, low_freqs, reverse_order, resolution, block_sz, Y_bound, path, n)
        )
        p.start()
        workers.append(p)

    for i in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
        # GPU generation
        samples = sample_fn(mini_batch_size)
        samples = samples.detach().cpu().numpy()

        # CPU queue push
        for b_id in range(mini_batch_size):
            img_id = i * mini_batch_size * world_size + local_rank * mini_batch_size + b_id
            if img_id >= n_samples:
                break
            sample_queue.put((img_id, samples[b_id]))

    # Finish: Wait for queue to drain and terminate workers
    stop_event.set()
    while not sample_queue.empty():
        pass
    for p in workers:
        p.join()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'generated {len(os.listdir(path))} images...')
        assert len(os.listdir(path)) == n_samples


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def images_to_npz(directory, output_file):
    images_list = []  # List to store image arrays

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
            file_path = os.path.join(directory, filename)

            with Image.open(file_path) as img:
                image_array = np.array(img)
                images_list.append(image_array)

    if images_list:
        all_images_array = np.stack(images_list, axis=0)
        np.savez(output_file, all_images_array)
        print(f"All images have been saved to {output_file} with shape {all_images_array.shape}")
    else:
        print("No images to save.")