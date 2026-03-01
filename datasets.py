from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import cv2
from DCT_utils import *


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

# EchoNet-Dynamic Dataset

class EchoNetUncond(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz
        self.Y_bound = kwargs.get('Y_bound', None)
        # transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        # transforms.Normalize(0.5, 0.5)])
        self.greyscale = kwargs.get('greyscale', False)
        self.dataset_type = kwargs.get('dataset_type', 'dynamic')
        self.frequency_aware_tokens = kwargs.get('frequency_aware_tokens', False)

        if self.greyscale:
            if self.frequency_aware_tokens:
                self.num_fa_repeats = kwargs.get('num_fa_repeats', 4)
                self.num_fa_length = kwargs.get('num_fa_length', 0)
                self.num_fa_repeats_x = kwargs.get('num_fa_repeats_x', 0)
                self.num_fa_repeats_y = kwargs.get('num_fa_repeats_y', 0)
                assert self.num_fa_length > 0 and self.num_fa_repeats > 0, "num_fa_length and num_fa_repeatsmust be > 0 for frequency aware tokens."
                self.train = DCT_FA_Customized(
                    data_property={'mean': kwargs.get('Y_mean', None), 'std': kwargs.get('Y_std', None),'min': kwargs.get('Y_min', None), 'max': kwargs.get('Y_max', None), 'Y_bound': self.Y_bound},
                    path=path, img_sz=resolution, low_freqs=low_freqs, block_sz=block_sz, num_fa_length=self.num_fa_length, num_fa_repeats=self.num_fa_repeats, tokenwise_normalization=kwargs.get('tokenwise_normalization', 'Y_bound'), num_fa_repeats_x=self.num_fa_repeats_x, num_fa_repeats_y=self.num_fa_repeats_y
                )
                self.block_component = None  # will be determined by num_fa_length

            else:
                self.block_component = 4  # only Y channel
                self.train = DCT_4Y(
                    path=path, img_sz=resolution, tokens=tokens,
                    low_freqs=low_freqs, block_sz=block_sz, Y_bound=self.Y_bound, cache=kwargs.get('cache', False), cache_name=kwargs.get('cache_name', 'echonet_cache')
                )

        else:
            self.block_component = 6  # Y-Cb-Cr 
            self.train = DCT_4YCbCr(
                path=path, img_sz=resolution, tokens=tokens,
                low_freqs=low_freqs, block_sz=block_sz, Y_bound=self.Y_bound
            )

    @property
    def data_shape(self):
        if self.frequency_aware_tokens:
            return self.tokens, self.num_fa_length * self.num_fa_repeats
        else: 
            return self.tokens, self.low_freqs*self.block_component

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/echo.npz'
        
    @property
    def has_label(self):
        return False



# ACDC Unlabeled Dataset

class ACDCUncond(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz
        self.Y_bound = kwargs.get('Y_bound', None)
        # transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        # transforms.Normalize(0.5, 0.5)])
        self.greyscale = kwargs.get('greyscale', False)
        self.dataset_type = kwargs.get('dataset_type', 'full')
        self.positional_tokens = kwargs.get('positional_tokens', False)
        self.frequency_aware_tokens = kwargs.get('frequency_aware_tokens', False)
        assert not (self.positional_tokens and self.frequency_aware_tokens), "Cannot use both positional tokens and frequency aware tokens."

        if self.greyscale:
            if self.positional_tokens:
                assert kwargs['tokenwise_normalization'] in ['z-score', 'minmax'], "If using positional_tokens, token-wise normalization must be provided."
                self.tokenwise_normalization = kwargs['tokenwise_normalization']
                if self.tokenwise_normalization == 'z-score':
                    assert 'Y_mean' in kwargs.keys() and 'Y_std' in kwargs.keys(), "Mean and std must be provided for positional token normalization."
                    mean = kwargs['Y_mean']
                    std = kwargs['Y_std']
                    data_property = {'mean': mean, 'std': std}
                elif self.tokenwise_normalization == 'minmax':
                    assert 'Y_min' in kwargs.keys() and 'Y_max' in kwargs.keys(), "Min and max must be provided for positional token normalization."
                    _min = kwargs['Y_min']
                    _max = kwargs['Y_max']
                    data_property = {'min': _min, 'max': _max}
                else:
                    raise NotImplementedError("Only 'z-score' and 'minmax' normalization are supported for positional tokens.")
                
                print("Using DCT_PositionalToken dataset with token-wise normalization.")
                self.train = DCT_PositionalToken(
                    data_property=data_property,
                    path=path, img_sz=resolution, low_freqs=low_freqs,
                    block_sz=block_sz, normalization=self.tokenwise_normalization
                )
                self.tokens = low_freqs  # In positional token setting, number of tokens equals low_freqs
                self.block_component = 1  # only Y channel

            elif self.frequency_aware_tokens:
                self.num_fa_repeats = kwargs.get('num_fa_repeats', 4)
                self.num_fa_length = kwargs.get('num_fa_length', 0)
                self.num_fa_repeats_x = kwargs.get('num_fa_repeats_x', 0)
                self.num_fa_repeats_y = kwargs.get('num_fa_repeats_y', 0)
                assert self.num_fa_length > 0 and self.num_fa_repeats > 0, "num_fa_length and num_fa_repeatsmust be > 0 for frequency aware tokens."
                self.train = DCT_FA_Customized(
                    data_property={'mean': kwargs.get('Y_mean', None), 'std': kwargs.get('Y_std', None),'min': kwargs.get('Y_min', None), 'max': kwargs.get('Y_max', None), 'Y_bound': self.Y_bound},
                    path=path, img_sz=resolution, low_freqs=low_freqs, block_sz=block_sz, num_fa_length=self.num_fa_length, num_fa_repeats=self.num_fa_repeats, tokenwise_normalization=kwargs.get('tokenwise_normalization', 'Y_bound'), num_fa_repeats_x=self.num_fa_repeats_x, num_fa_repeats_y=self.num_fa_repeats_y
                )
                self.block_component = None  # will be determined by num_fa_length

            else:
                self.block_component = 4  # only Y channel
                self.train = DCT_4Y(
                    path=path, img_sz=resolution, tokens=tokens,
                    low_freqs=low_freqs, block_sz=block_sz, Y_bound=self.Y_bound, cache=kwargs.get('cache', False), cache_name=kwargs.get('cache_name', 'acdc_uncond_wholeheart_4Y')
                )

        else:
            self.block_component = 6  # Y-Cb-Cr 
            self.train = DCT_4YCbCr(
                path=path, img_sz=resolution, tokens=tokens,
                low_freqs=low_freqs, block_sz=block_sz, Y_bound=self.Y_bound
            )

    @property
    def data_shape(self):
        if self.frequency_aware_tokens:
            return self.tokens, self.num_fa_length * self.num_fa_repeats
        elif self.positional_tokens:
            return self.tokens, self.resolution*self.resolution // (self.block_sz * self.block_sz)  # (low_freqs, num_blocks)
        else: 
            return self.tokens, self.low_freqs*self.block_component

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        if self.greyscale:
            if self.dataset_type == 'compact':
                return 'data/scratch/U-ViT2/assets/fid_stats/acdc_unlabel_compact_greyscale.npz'
            elif self.dataset_type == 'wholeheart':
                return 'data/scratch/U-ViT2/assets/fid_stats/acdc_unlabel_wholeheart_greyscale.npz'
            else:
                return 'data/scratch/U-ViT2/assets/fid_stats/acdc_unlabel_greyscale.npz'
        else:
            return 'data/scratch/U-ViT2/assets/fid_stats/acdc_unlabel.npz'
        
    @property
    def has_label(self):
        return False

# ACDC Uncond Images Dataset
class UnlabeledImageDataset(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.img_paths = _list_image_files_recursively(path)
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('RGB')
        img = self.transform(img)
        img = img * 2.0 - 1.0  # scale to [-1, 1]
        return img

class ACDCUncondImages(DatasetFactory):
    def __init__(self, path, resolution=96, **kwargs):
        super().__init__()

        self.resolution = resolution
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        self.train = UnlabeledImageDataset(path=path, transform=transform)

    @property
    def data_shape(self):
        return 1, self.resolution, self.resolution  # greyscale images

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/acdc_unlabel_wholeheart_greyscale.npz'
        
    @property
    def has_label(self):
        return False

    def unpreprocess(self, v):
        return super().unpreprocess(v)

class EchoNetUncondImages(DatasetFactory):
    def __init__(self, path, resolution=112, **kwargs):
        super().__init__()

        self.resolution = resolution
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        self.train = UnlabeledImageDataset(path=path, transform=transform)

    @property
    def data_shape(self):
        return 1, self.resolution, self.resolution  # greyscale images

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/echo.npz'
        
    @property
    def has_label(self):
        return False

    def unpreprocess(self, v):
        return super().unpreprocess(v)
# ACDC labeled Dataset

class ACDCCond(DatasetFactory):
    def __init__(self, path:tuple, resolution:int=96, tokens:int=144, low_freqs:int=16, block_sz:int=4, Y_bound:int=1, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs
        
        if 'greyscale' in kwargs.keys():
            self.greyscale = kwargs['greyscale']
        else:
            self.greyscale = False
            
        if self.greyscale:
            self.block_component = 4  # only Y channel
            self.train = DCT_4Y_Cond(
                img_dir=path[0],
                label_dir=path[1],
                img_sz=resolution, 
                tokens=tokens,
                low_freqs=low_freqs, 
                block_sz=block_sz, 
                Y_bound=Y_bound
            )
        else:
            raise NotImplementedError("ACDCCond dataset only supports greyscale images currently.")

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*self.block_component  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training        
        return 'data/scratch/U-ViT2/assets/fid_stats/acdc_labeled_greyscale.npz'

        
    @property
    def has_label(self):
        return True
    
    def sample_label(self, n_samples, device):
        """
        Sample labels for the dataset.
        :param n_samples: number of samples to generate
        :param device: device to place the labels on
        :return: tensor of sampled labels
        """
        return self.train.sample_label(n_samples, device)


# CIFAR10

class CIFAR10(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = DCT_4YCbCr(
            path=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/fid_stats_cifar10_train.npz'

    @property
    def has_label(self):
        return False



# ImageNet


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z, label = np.load(path, allow_pickle=True)
        return z, label


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            if entry.startswith('cache'):
                continue
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class DCT_4YCbCr(Dataset):
    def __init__(self, path, img_sz=64, tokens=0, low_freqs=0, block_sz=8, Y_bound=None):
        self.path = path
        self.classes = os.listdir(path)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(path, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using eta {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = zigzag_order(block_sz)
        self.reverse_order = reverse_zigzag_order(block_sz)

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img)

        # Step 1: Convert RGB to YCbCr
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img_y = 0.299 * R + 0.587 * G + 0.114 * B
        img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)
        cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 2: Split the Y, Cb, and Cr components into BxB blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (h, w) --> (h/B * w/B, B, B)
        cb_blocks = split_into_blocks(cb_downsampled, self.block_sz)  # Cb component, (h/2, w/2) --> (h/2B * w/2B, B, B)
        cr_blocks = split_into_blocks(cr_downsampled, self.block_sz)  # Cr component, (h/2, w/2) --> (h/2B * w/2B, B, B)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (h/B * w/B, B, B)
        dct_cb_blocks = dct_transform(cb_blocks)  # (h/2B * w/2B, B, B)
        dct_cr_blocks = dct_transform(cr_blocks)  # (h/2B * w/2B, B, B)

        # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks region)
        DCT_blocks = []
        for i in range(dct_cr_blocks.shape[0]):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],  # Y
                dct_y_blocks[self.index[i] + 1],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],  # Y
                dct_cb_blocks[i],  # Cb
                dct_cr_blocks[i],  # Cr
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz*self.block_sz)  # (num_tokens, 6, B**2)

        # Step 5: scale into [-1, 1]
        assert DCT_blocks.shape == (self.tokens, 6, self.block_sz*self.block_sz)
        DCT_blocks[:, :4 :] = (DCT_blocks[:, :4 :]) / self.Y_bound
        DCT_blocks[:, 4, :] = (DCT_blocks[:, 4, :]) / self.Y_bound
        DCT_blocks[:, 5, :] = (DCT_blocks[:, 5, :]) / self.Y_bound

        # Step 6: reorder coe from low to high freq, then mask out high-freq signals
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (num_tokens, 6, B**2)
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (num_tokens, 6, B**2) --> (num_tokens, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (num_tokens, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32

        return DCT_blocks

class DCT_4Y(Dataset):
    def __init__(self, path, img_sz=64, tokens=0, low_freqs=0, block_sz=8, Y_bound=None, cache=False, cache_name='Undefined'):
        self.path = path
        self.img_paths = _list_image_files_recursively(path)

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using eta {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):
            for col in range(0, self.Y_blocks_per_row, 2):
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = zigzag_order(block_sz)
        self.reverse_order = reverse_zigzag_order(block_sz)
        self.cache = cache
        if self.cache:
            self.cache_path = os.path.join(path, f'{cache_name}')
            if os.path.exists(self.cache_path):
                print(f'Loading cached DCT_4Y data from {self.cache_path}...')
            else:
                print(f'Cache data {self.cache_path} not found. Skipping cache...')
                self.cache = False

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.cache:
            cached_file = os.path.join(self.cache_path, f'{idx}.pt')
            DCT_blocks = torch.load(cached_file, weights_only=True)
            return DCT_blocks
        else:
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('L')  # 灰度图
            img = transforms.RandomHorizontalFlip()(img)
            img = np.array(img)

            # Step 1: Y channel就是灰度图本身
            img_y = img.astype(np.float32)

            # Step 2: Split Y into BxB blocks
            y_blocks = split_into_blocks(img_y, self.block_sz)  # (h, w) --> (h/B * w/B, B, B)

            # Step 3: Apply DCT on each block
            dct_y_blocks = dct_transform(y_blocks)  # (num_blocks, B, B)

            # Step 4: 组织token顺序
            DCT_blocks = []
            for i in range(dct_y_blocks.shape[0]// 4):
                DCT_blocks.append([
                    dct_y_blocks[self.index[i]],
                    dct_y_blocks[self.index[i] + 1],
                    dct_y_blocks[self.index[i] + self.Y_blocks_per_row],
                    dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],
                ])
            DCT_blocks = np.array(DCT_blocks).reshape(-1, 4, self.block_sz * self.block_sz)  # (tokens, 4, B**2)

            # Step 5: scale into [-1, 1]
            assert DCT_blocks.shape == (self.tokens, 4, self.block_sz * self.block_sz)
            DCT_blocks = DCT_blocks / self.Y_bound  # 广播

            # Step 6: zigzag排序+mask高频
            DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (tokens, 4, B**2)
            DCT_blocks = DCT_blocks[:, :, :self.low_freqs]      # (tokens, 4, low_freqs)

            # numpy to torch
            DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (tokens, 4*low_freqs)
            DCT_blocks = DCT_blocks.float()

            return DCT_blocks

class DCT_4Y_FA(Dataset):
    """decrepated"""
    def __init__(self, path, img_sz=64, tokens=0, low_freqs=0, block_sz=8, Y_bound=None, **kwargs):
        self.path = path
        self.classes = os.listdir(path)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_fa_length = kwargs.get('num_fa_length', 2)
        assert low_freqs % self.num_fa_length == 0, "low_freqs must be divisible by num_fa_length."
        self.img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(path, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using eta {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):
            for col in range(0, self.Y_blocks_per_row, 2):
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = zigzag_order(block_sz)
        self.reverse_order = reverse_zigzag_order(block_sz)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = Image.open(img_path).convert('L')  # 灰度图
        img = transforms.RandomHorizontalFlip()(img)
        img = np.array(img)

        # Step 1: Y channel就是灰度图本身
        img_y = img.astype(np.float32)

        # Step 2: Split Y into BxB blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # (h, w) --> (h/B * w/B, B, B)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (num_blocks, B, B)

        # Step 4: 组织token顺序
        DCT_blocks = []
        for i in range(dct_y_blocks.shape[0]// 4):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],
                dct_y_blocks[self.index[i] + 1],
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 4, self.block_sz * self.block_sz)  # (tokens, 4, B**2)

        # Step 5: scale into [-1, 1]
        # assert DCT_blocks.shape == (self.tokens, 4, self.block_sz * self.block_sz)
        if len(self.Y_bound) == 2:
            DCT_blocks = (DCT_blocks - np.min(self.Y_bound)) / (np.max(self.Y_bound) - np.min(self.Y_bound)) * 2 - 1  # 广播
        else:
            DCT_blocks = DCT_blocks / self.Y_bound  # 广播

        # Step 6: zigzag排序+mask高频
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (tokens, 4, B**2)
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]      # (tokens, 4, low_freqs)
        DCT_blocks_Pos = []
        num_per_freq = self.low_freqs // self.num_fa_length
        for i in range(self.num_fa_length):
            freq_pos = DCT_blocks[:, :, i * num_per_freq:(i + 1) * num_per_freq]  # (tokens, 4, low_freqs//num_fa_length)
            freq_pos = freq_pos.reshape(-1, 4 * num_per_freq)  # (tokens, 4*low_freqs//num_fa_length)
            DCT_blocks_Pos.append(freq_pos)
        DCT_blocks_Pos = np.concatenate(DCT_blocks_Pos, axis=0) # (tokens*num_fa_length, 4*low_freqs//num_fa_length)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks_Pos).float()  # (tokens*num_fa_length, 4*low_freqs/num_fa_length)

        return DCT_blocks

    def reverse_ordering(self, tokens):
        """
        Reverse the ordering of DCT coefficients from low-to-high frequency back to
        the original block order.
        :param tokens: Tensor of shape (tokens*num_fa_length, 4*low_freqs/num_fa_length)
        :return: Tensor of shape (tokens, 4*low_freqs*low_freqs) with original ordering
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()

        if len(tokens.shape) == 3:
            stack_output = []
            for i in range(tokens.shape[0]):
                stack_output.append(self.reverse_ordering(tokens[i]))
            return torch.stack(stack_output, dim=0)
        
        num_per_freq = self.low_freqs // self.num_fa_length
        reverse_order = []
        for i in range(self.num_fa_length):
            temp = tokens[ i * tokens.shape[0] // self.num_fa_length : (i + 1) * tokens.shape[0] // self.num_fa_length, :]
            temp = temp.reshape(-1, 4, num_per_freq)  # (tokens, 4, low_freqs//num_fa_length)
            reverse_order.append(temp)
        reverse_order = np.concatenate(reverse_order, axis=2)  # (tokens, 4, low_freqs)

        return torch.from_numpy(reverse_order).reshape(-1, 4 * self.low_freqs * self.low_freqs).float()  # (num_tokens, 4*block_sz*block_sz)

class DCT_FA_Customized(Dataset):
    """
    Dataset class that processes greyscale images into DCT tokens.
    Each token consists of DCT coefficients from the same frequency positions
    across all blocks in an image. The tokens are ordered in zigzag scan order,
    and high-frequency tokens are truncated based on `low_freqs`.
    The tokens are then normalized using a provided mean and std.
    """
    def __init__(self, data_property:dict, path, img_sz=96, low_freqs=16, block_sz=4, tokenwise_normalization='z-score', **kwargs):
        self.path = path
        # Assuming a flat directory of images for simplicity
        self.img_paths = _list_image_files_recursively(path)
        print(f"Found {len(self.img_paths)} images in {path}")
        assert tokenwise_normalization in ['z-score','minmax','Y_bound'], "Normalization must be either 'z-score' or 'minmax' or 'Y_bound'."
        self.normalization = tokenwise_normalization
        print(f"Using {self.normalization} normalization for tokens.")
        self.num_fa_length = kwargs.get('num_fa_length', 2)
        self.num_fa_repeats = kwargs.get('num_fa_repeats', 4)
        self.num_fa_repeats_x = kwargs.get('num_fa_repeats_x', None)
        self.num_fa_repeats_y = kwargs.get('num_fa_repeats_y', None)
        print(f"Using num_fa_length={self.num_fa_length}, num_fa_repeats={self.num_fa_repeats, self.num_fa_repeats_x, self.num_fa_repeats_y} for frequency-aware tokenization.")
        assert low_freqs % self.num_fa_length == 0, "low_freqs must be divisible by num_fa_length."

        if self.normalization == 'z-score': 
            assert 'mean' in data_property.keys() and 'std' in data_property.keys(), "Mean and std must be provided for z-score normalization."
            mean = data_property['mean']
            std = data_property['std']
            # Parameters for DCT processing
            assert mean is not None and std is not None, "Mean and std must be provided for normalization."
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
            print(f"Using mean and std for token-wise normalization")

        elif self.normalization == 'minmax':
            assert 'min' in data_property.keys() and 'max' in data_property.keys(), "Min and max must be provided for min-max normalization."
            _min = data_property['min']
            _max = data_property['max']
            assert _min is not None and _max is not None, "Min and max must be provided for normalization."
            self.min = np.array(_min, dtype=np.float32)
            self.max = np.array(_max, dtype=np.float32)
            print(f"Using min and max for token-wise normalization")

        elif self.normalization == 'Y_bound':
            self.y_bound = np.array(data_property['Y_bound'], dtype=np.float32)
            print(f"Using Y_bound={self.y_bound} for scaling")

        self.low_freqs = low_freqs
        self.block_sz = block_sz
        self.img_sz = img_sz

        # The number of blocks determines the length of each token
        self.num_blocks = (img_sz // block_sz) ** 2
        assert self.num_blocks % self.num_fa_repeats == 0, "The number of blocks must be divisible by num_fa_repeats."
        # The number of positions in a block determines the total number of tokens before truncation
        self.num_positions = block_sz ** 2
        assert self.low_freqs <= self.num_positions, "low_freqs cannot be greater than the number of DCT coefficients in a block."
        # assert len(self.mean) >= self.low_freqs and len(self.std) >= self.low_freqs, "Mean and std arrays must have at least low_freqs elements."

        # Zigzag order to arrange tokens by frequency
        self.low2high_order = zigzag_order(block_sz)
        self.macroblock_forward_indices, self.macroblock_reverse_indices = get_macroblock_indices(
            image_shape=(self.img_sz, self.img_sz),
            block_sz=self.block_sz,
            x=self.num_fa_repeats_x,
            y=self.num_fa_repeats_y
        )

        self.cache = kwargs.get('cache', True)
        if self.cache:
            self.cache_dir = os.path.join(path, f'cache_dct_fa_{block_sz}by{block_sz}_low{low_freqs}_l{self.num_fa_length}r{self.num_fa_repeats}_{self.normalization}')
            if os.path.exists(self.cache_dir):
                print(f"Loading cached DCT tokens from {self.cache_dir}")
            else:
                print(f'Cache directory {self.cache_dir} does not exist. Caching will be skipped.')
                self.cache = False

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.cache:
            cached_file = os.path.join(self.cache_dir, f'{idx}.pt')
            DCT_fa_tokens = torch.load(cached_file, weights_only=True)
        else:
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('L')  # Convert to greyscale
            img = transforms.RandomHorizontalFlip()(img)
            img = np.array(img, dtype=np.float32)

            # Step 1: Split the Y channel (the greyscale image itself) into blocks
            y_blocks = split_into_blocks(img, self.block_sz)  # Shape: (num_blocks, block_sz, block_sz)

            # Step 2: Apply DCT to each block
            dct_y_blocks = dct_transform(y_blocks)  # Shape: (num_blocks, block_sz, block_sz)

            # Step 3: Reshape and transpose to group coefficients by position
            positional_tokens = dct_y_blocks.reshape(self.num_blocks, self.num_positions) # (num_blocks, num_positions)
            # positional_tokens = flattened_dct.T # Shape: (num_positions, num_blocks)

            # Step 4: Order tokens by frequency using zigzag order
            ordered_tokens = positional_tokens[:, self.low2high_order] # Shape: (num_blocks, num_positions)

            # Step 5: Truncate high-frequency tokens using `low_freqs`
            final_tokens = ordered_tokens[:, :self.low_freqs] # Shape: (num_blocks, low_freqs)

            # Step 6: Normalize each token (frequency) using the provided mean and std.
            if self.normalization == 'z-score':
                mean = self.mean[:self.low_freqs].reshape(-1)
                std = self.std[:self.low_freqs].reshape(-1)
                # Add a small epsilon to std to avoid division by zero
                normalized_tokens = (final_tokens - mean) / (std + 1e-8)
            elif self.normalization == 'minmax':
                _min = self.min[:self.low_freqs].reshape(-1)
                _max = self.max[:self.low_freqs].reshape(-1)
                normalized_tokens = 2 * (final_tokens - _min) / (_max - _min + 1e-8) - 1  # Scale to [-1, 1]
            elif self.normalization == 'Y_bound':
                # normalized_tokens = final_tokens / self.y_bound * 2 - 1 # Scale to approximately [-1, 1]
                normalized_tokens = final_tokens / self.y_bound

            # step 6.5: select frequency-aware tokens by repeating separate DCT coefficients using num_fa_repeats and num_fa_length
            # normalized_tokens = normalized_tokens.T  # (low_freqs, num_blocks) --> (num_blocks, low_freqs)
            normalized_tokens = normalized_tokens[self.macroblock_forward_indices, :]  # (num_blocks, low_freqs)
            DCT_fa_tokens = self.FA_transform(normalized_tokens)  # (num_blocks//num_fa_repeats * low_freqs//num_fa_length, num_fa_repeats * num_fa_length)

            # Step 7: Convert to a FloatTensor
        return DCT_fa_tokens
    
    def FA_transform(self, normalized_tokens, entropy_transform=False):
        """
        Transform the normalized postional tokens into frequency-aware tokens.
        :param normalized_tokens: A numpy array of shape (num_blocks, low_freqs).
        :return: A numpy array of shape (num_blocks//num_fa_repeats * low_freqs//num_fa_length, num_fa_repeats * num_fa_length).
        """
        if entropy_transform:
            normalized_tokens = torch.stack(self.num_blocks * [normalized_tokens], dim=0)  # (num_blocks, low_freqs)

        if isinstance(normalized_tokens, torch.Tensor):
            normalized_tokens = normalized_tokens.detach().cpu().numpy()

        DCT_fa_tokens = []
        num_tokens_per_block = self.low_freqs // self.num_fa_length
        num_blocks_per_token = self.num_blocks // self.num_fa_repeats
        
        for i in range(num_blocks_per_token):
            block_freq = normalized_tokens[i * self.num_fa_repeats : (i + 1) * self.num_fa_repeats, :]  # (num_fa_repeats, low_freqs)
            all_freq_tokens = []
            for j in range(num_tokens_per_block):
                freq_token = block_freq[:, j * self.num_fa_length : (j + 1) * self.num_fa_length]  # (num_fa_repeats, num_fa_length)
                freq_token = freq_token.reshape(-1)  # Flatten to (num_fa_repeats * num_fa_length,)
                all_freq_tokens.append(freq_token)
            all_freq_tokens = np.stack(all_freq_tokens, axis=0)  # (num_tokens_per_block, num_fa_repeats * num_fa_length)
            DCT_fa_tokens.append(all_freq_tokens)
        DCT_fa_tokens = np.concatenate(DCT_fa_tokens, axis=0)  # (num_blocks_per_token * num_tokens_per_block, num_fa_repeats * num_fa_length)
        return torch.from_numpy(DCT_fa_tokens).float()

    def denormalize(self, normalized_tokens):
        """
        Denormalizes a tensor of tokens back to the original DCT coefficient scale.
        :param normalized_tokens: A tensor of shape (B, low_freqs, num_blocks) or (low_freqs, num_blocks).
        :return: A tensor with the same shape, in the original DCT scale.
        """
        if self.normalization == 'z-score':
            # Ensure mean and std are on the correct device and have the correct shape for broadcasting
            mean = torch.from_numpy(self.mean[:self.low_freqs]).to(normalized_tokens.device)
            std = torch.from_numpy(self.std[:self.low_freqs]).to(normalized_tokens.device)
            
            # Reshape for broadcasting over batches and tokens
            # Adds a batch dimension if the input is a single sample
            if normalized_tokens.dim() == 2:
                normalized_tokens = normalized_tokens.unsqueeze(0) # (low_freqs, num_blocks) -> (1, low_freqs, num_blocks)
            
            # Reshape mean and std to (1, low_freqs, 1) to broadcast across batch and block dimensions
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)

            # Apply denormalization: value = (normalized_value * std) + mean
            denormalized_tokens = normalized_tokens * (std + 1e-8) + mean
        elif self.normalization == 'minmax':
            _min = torch.from_numpy(self.min[:self.low_freqs]).to(normalized_tokens.device)
            _max = torch.from_numpy(self.max[:self.low_freqs]).to(normalized_tokens.device)

            if normalized_tokens.dim() == 2:
                normalized_tokens = normalized_tokens.unsqueeze(0) # (low_freqs, num_blocks) -> (1, low_freqs, num_blocks)

            _min = _min.view(1, -1, 1)
            _max = _max.view(1, -1, 1)

            # Apply denormalization: value = ((normalized_value + 1) / 2) * (max - min) + min
            denormalized_tokens = ((normalized_tokens + 1) / 2) * (_max - _min + 1e-8) + _min
        
        elif self.normalization == 'Y_bound':
            # denormalized_tokens = (normalized_tokens + 1) * self.y_bound / 2
            denormalized_tokens = normalized_tokens * self.y_bound

        return denormalized_tokens.squeeze(0) # Remove batch dim if it was added
    
    def reverse_ordering(self, tokens):
        """
        Reverse the DCT_fa_Customized ordering of DCT coefficients from frequency-aware tokens to (low_freqs, num_blocks).
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()

        if len(tokens.shape) == 3:
            stack_output = []
            for i in range(tokens.shape[0]):
                stack_output.append(self.reverse_ordering(tokens[i]))
            return torch.stack(stack_output, dim=0)
        
        num_tokens_per_block = self.low_freqs // self.num_fa_length
        num_blocks_per_token = self.num_blocks // self.num_fa_repeats
        
        reverse_order = []
        for i in range(num_blocks_per_token):
            block_freq = tokens[i * num_tokens_per_block : (i + 1) * num_tokens_per_block, :]  # (num_tokens_per_block, num_fa_repeats * num_fa_length)
            all_freqs = []
            for j in range(num_tokens_per_block):
                freq = block_freq[j, :]  # (, num_fa_repeats * num_fa_length)
                freq = freq.reshape(self.num_fa_repeats, self.num_fa_length)  # (num_fa_repeats, num_fa_length)
                all_freqs.append(freq)
            all_freqs = np.concatenate(all_freqs, axis=1)  # (num_fa_repeats, num_fa_length) --> (num_fa_repeats, low_freqs)
            reverse_order.append(all_freqs)
        reverse_order = np.concatenate(reverse_order, axis=0)  # (num_fa_repeats, low_freqs) --> (num_blocks, low_freqs)
        reverse_order = reverse_order[self.macroblock_reverse_indices, :]  # (num_blocks, low_freqs)
        return torch.from_numpy(reverse_order).T.float()  # (low_freqs, num_blocks)

class DCT_PositionalToken(Dataset):
    """
    Dataset class that processes greyscale images into DCT tokens.
    Each token consists of DCT coefficients from the same frequency position
    across all blocks in an image. The tokens are ordered in zigzag scan order,
    and high-frequency tokens are truncated based on `low_freqs`.
    The tokens are then normalized using a provided mean and std.
    """
    def __init__(self, data_property:dict, path, img_sz=96, low_freqs=16, block_sz=4, normalization='z-score', **kwargs):
        self.path = path
        # Assuming a flat directory of images for simplicity
        self.img_paths = _list_image_files_recursively(path)
        print(f"Found {len(self.img_paths)} images in {path}")
        assert normalization in ['z-score','minmax'], "Normalization must be either 'z-score' or 'minmax'."
        self.normalization = normalization
        print(f"Using {self.normalization} normalization for tokens.")

        if self.normalization == 'z-score':
            assert 'mean' in data_property.keys() and 'std' in data_property.keys(), "Mean and std must be provided for z-score normalization."
            mean = data_property['mean']
            std = data_property['std']
            # Parameters for DCT processing
            assert mean is not None and std is not None, "Mean and std must be provided for normalization."
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
            print(f"Using mean and std for token-wise normalization")

        elif self.normalization == 'minmax':
            assert 'min' in data_property.keys() and 'max' in data_property.keys(), "Min and max must be provided for min-max normalization."
            _min = data_property['min']
            _max = data_property['max']
            assert _min is not None and _max is not None, "Min and max must be provided for normalization."
            self.min = np.array(_min, dtype=np.float32)
            self.max = np.array(_max, dtype=np.float32)
            print(f"Using min and max for token-wise normalization")

        elif self.normalization == 'Y_bound':
            self.y_bound = np.array(data_property['max'], dtype=np.float32).max()
            print(f"Using Y_bound={self.y_bound} for scaling")

        self.low_freqs = low_freqs
        self.block_sz = block_sz
        self.img_sz = img_sz

        # The number of blocks determines the length of each token
        self.num_blocks = (img_sz // block_sz) ** 2
        
        # The number of positions in a block determines the total number of tokens before truncation
        self.num_positions = block_sz ** 2
        assert self.low_freqs <= self.num_positions, "low_freqs cannot be greater than the number of DCT coefficients in a block."
        # assert len(self.mean) >= self.low_freqs and len(self.std) >= self.low_freqs, "Mean and std arrays must have at least low_freqs elements."

        # Zigzag order to arrange tokens by frequency
        self.low2high_order = zigzag_order(block_sz)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('L')  # Convert to greyscale
        img = transforms.RandomHorizontalFlip()(img)
        img = np.array(img, dtype=np.float32)

        # Step 1: Split the Y channel (the greyscale image itself) into blocks
        y_blocks = split_into_blocks(img, self.block_sz)  # Shape: (num_blocks, block_sz, block_sz)

        # Step 2: Apply DCT to each block
        dct_y_blocks = dct_transform(y_blocks)  # Shape: (num_blocks, block_sz, block_sz)

        # Step 3: Reshape and transpose to group coefficients by position
        flattened_dct = dct_y_blocks.reshape(self.num_blocks, self.num_positions)
        positional_tokens = flattened_dct.T # Shape: (num_positions, num_blocks)

        # Step 4: Order tokens by frequency using zigzag order
        ordered_tokens = positional_tokens[self.low2high_order, :] # Shape: (num_positions, num_blocks)

        # Step 5: Truncate high-frequency tokens using `low_freqs`
        final_tokens = ordered_tokens[:self.low_freqs, :] # Shape: (low_freqs, num_blocks)

        # Step 6: Normalize each token (frequency) using the provided mean and std.
        if self.normalization == 'z-score':
            mean = self.mean[:self.low_freqs].reshape(-1)
            std = self.std[:self.low_freqs].reshape(-1)
            # Add a small epsilon to std to avoid division by zero
            normalized_tokens = (final_tokens - mean) / (std + 1e-8)
        elif self.normalization == 'minmax':
            _min = self.min[:self.low_freqs].reshape(-1)
            _max = self.max[:self.low_freqs].reshape(-1)
            normalized_tokens = 2 * (final_tokens - _min) / (_max - _min + 1e-8) - 1  # Scale to [-1, 1]
        elif self.normalization == 'Y_bound':
            normalized_tokens = final_tokens / self.y_bound  # Scale to approximately [-1, 1]

        # Step 7: Convert to a FloatTensor
        return torch.from_numpy(normalized_tokens).float()

    def denormalize(self, normalized_tokens):
        """
        Denormalizes a tensor of tokens back to the original DCT coefficient scale.
        :param normalized_tokens: A tensor of shape (B, low_freqs, num_blocks) or (low_freqs, num_blocks).
        :return: A tensor with the same shape, in the original DCT scale.
        """
        if self.normalization == 'z-score':
            # Ensure mean and std are on the correct device and have the correct shape for broadcasting
            mean = torch.from_numpy(self.mean[:self.low_freqs]).to(normalized_tokens.device)
            std = torch.from_numpy(self.std[:self.low_freqs]).to(normalized_tokens.device)
            
            # Reshape for broadcasting over batches and tokens
            # Adds a batch dimension if the input is a single sample
            if normalized_tokens.dim() == 2:
                normalized_tokens = normalized_tokens.unsqueeze(0) # (low_freqs, num_blocks) -> (1, low_freqs, num_blocks)
            
            # Reshape mean and std to (1, low_freqs, 1) to broadcast across batch and block dimensions
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)

            # Apply denormalization: value = (normalized_value * std) + mean
            denormalized_tokens = normalized_tokens * (std + 1e-8) + mean
        elif self.normalization == 'minmax':
            _min = torch.from_numpy(self.min[:self.low_freqs]).to(normalized_tokens.device)
            _max = torch.from_numpy(self.max[:self.low_freqs]).to(normalized_tokens.device)

            if normalized_tokens.dim() == 2:
                normalized_tokens = normalized_tokens.unsqueeze(0) # (low_freqs, num_blocks) -> (1, low_freqs, num_blocks)

            _min = _min.view(1, -1, 1)
            _max = _max.view(1, -1, 1)

            # Apply denormalization: value = ((normalized_value + 1) / 2) * (max - min) + min
            denormalized_tokens = ((normalized_tokens + 1) / 2) * (_max - _min + 1e-8) + _min

        elif self.normalization == 'Y_bound':
            denormalized_tokens = (normalized_tokens + 1) * self.y_bound / 2
        
        return denormalized_tokens.squeeze(0) # Remove batch dim if it was added

class DCT_4Y_Cond(Dataset):
    def __init__(self, img_dir:str, label_dir:str, img_sz:int=96, tokens:int=144, low_freqs:int=16, block_sz:int=4, Y_bound:int=1):
        """
        :param img_dir: directory of images
        :param label_dir: directory of labels
        :param img_sz: size of images, e.g., (96, 96)
        :param tokens: number of tokens, e.g., (144, 144)
        :param low_freqs: number of low frequency coefficients, e.g., (16, 16)
        :param block_sz: size of blocks, e.g., (4, 4)
        :param Y_bound: scaling factor for Y channel, e.g., (1, 1)
        In tuple form, (Image, Label) order.
        """
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_sz = img_sz
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz
        self.Y_bound = Y_bound

        self.imgpair_paths = []
    
        for img_name in os.listdir(img_dir):
            basename = os.path.splitext(img_name)[0]
            label_name = basename + '_label.png'  # assuming labels are in png format
            self.imgpair_paths.append((os.path.join(img_dir, img_name), os.path.join(label_dir, label_name)))

        # parameters of DCT design
        print(f"using eta {self.Y_bound} for training Image and Label seperately.")
        self.Y_bound = np.array(Y_bound)
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        # Image DCT parameters
        Y_img = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row_img = int(img_sz / block_sz)
        self.index_img = []
        for row in range(0, Y_img, int(2 * self.Y_blocks_per_row_img)):
            for col in range(0, self.Y_blocks_per_row_img, 2):
                self.index_img.append(row + col)
        assert len(self.index_img) == int(Y_img / 4)

        self.low2high_order_img = zigzag_order(block_sz)
        self.reverse_order_img = reverse_zigzag_order(block_sz)

        # Label DCT parameters
        Y_label = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row_label = int(img_sz / block_sz)
        self.index_label = []
        for row in range(0, Y_label, int(2 * self.Y_blocks_per_row_label)):
            for col in range(0, self.Y_blocks_per_row_label, 2):
                self.index_label.append(row + col)
        assert len(self.index_label) == int(Y_label / 4)

        self.low2high_order_label = zigzag_order(block_sz)
        self.reverse_order_label = reverse_zigzag_order(block_sz)

        self.label_class_num = np.unique(np.array(Image.open(self.imgpair_paths[0][1]).convert('L'))).shape[0]

    def label_recalibrate(self, label: np.ndarray) -> np.ndarray:
        """
        Assume you need to change numerical labels to match pixel space values or vice versa.
        Eg. labels in [0,1,2,3] need to be changed to [0, 85, 170, 255] for 8-bit grayscale.
        """
        if np.max(label) == 255:
            label = (label / 255 * self.label_class_num).astype(np.uint8)
        elif np.max(label) < self.label_class_num:
            label = (label * 255 / np.max(label)).astype(np.uint8)
        else:
            raise ValueError(f"Label values {np.unique(label)} do not match expected range for {self.label_class_num} classes.")
        return label

    def __len__(self):
        return len(self.imgpair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.imgpair_paths[idx]
        img = Image.open(img_path).convert('L')  # 灰度图
        label = Image.open(label_path).convert('L')  # 灰度图
        img = np.array(img)
        label = np.array(label)
        # label = self.label_recalibrate(label)  # Recalibrate label values if necessary

        img_DCT_blocks = self.get_DCT_blocks(img)
        label_DCT_blocks = self.get_DCT_blocks(label)

        return {'image':img_DCT_blocks, 'label':label_DCT_blocks}
    
    def sample_label(self, n_samples, device):
        """
        Sample labels for the dataset.
        :param n_samples: number of samples to generate
        :param device: device to place the labels on
        :return: tensor of sampled labels
        """
        rand_inds = torch.randint(0, len(self.imgpair_paths),(n_samples,))
        label_pths = [self.imgpair_paths[idx][1] for idx in rand_inds]
        labels = [Image.open(label_path).convert('L') for label_path in label_pths]
        labels_ready = []
        for label in labels:
            # label = self.label_recalibrate(np.array(label))
            labels_ready.append(np.array(label))

        label_dct_blocks = []
        # Get DCT blocks for each label
        for label in labels_ready:
            label_dct_block = self.get_DCT_blocks(label).to(device)  # (tokens, 4*low_freqs)
            label_dct_blocks.append((label_dct_block))
        return torch.stack(label_dct_blocks, dim=0)  # (n_samples, tokens, 4*low_freqs)
        
    def get_DCT_blocks(self,img: np.ndarray) -> torch.Tensor:
        # Step 1: Y channel就是灰度图本身
        img_y = img.astype(np.float32)

        # Step 2: Split Y into BxB blocks
        img_y_blocks = split_into_blocks(img_y, self.block_sz)  # (h, w) --> (h/B * w/B, B, B)

        # Step 3: Apply DCT on each block
        img_dct_y_blocks = dct_transform(img_y_blocks)  # (num_blocks, B, B)

        # Step 4: 组织token顺序
        img_DCT_blocks = []
        for i in range(img_dct_y_blocks.shape[0]// 4):
            img_DCT_blocks.append([
                img_dct_y_blocks[self.index_img[i]],
                img_dct_y_blocks[self.index_img[i] + 1],
                img_dct_y_blocks[self.index_img[i] + self.Y_blocks_per_row_img],
                img_dct_y_blocks[self.index_img[i] + self.Y_blocks_per_row_img + 1],
            ])
        img_DCT_blocks = np.array(img_DCT_blocks).reshape(-1, 4, self.block_sz * self.block_sz)  # (tokens, 4, B**2)
        
        # Step 5: scale into [-1, 1]
        assert img_DCT_blocks.shape == (self.tokens, 4, self.block_sz * self.block_sz)
        img_DCT_blocks = img_DCT_blocks / self.Y_bound  # 广播

        # Step 6: zigzag排序+mask高频
        img_DCT_blocks = img_DCT_blocks[:, :, self.low2high_order_img]  # (tokens, 4, B**2)
        img_DCT_blocks = img_DCT_blocks[:, :, :self.low_freqs]      # (tokens, 4, low_freqs)

        # numpy to torch
        img_DCT_blocks = torch.from_numpy(img_DCT_blocks).reshape(self.tokens, -1)  # (tokens, 4*low_freqs)
        img_DCT_blocks = img_DCT_blocks.float()
        return img_DCT_blocks

class DCT_4YCbCr_cond(Dataset):
    def __init__(self, img_sz=64, tokens=0, low_freqs=0, block_sz=8, train_files=None, labels=None, Y_bound=None):

        self.image_paths = train_files
        self.labels = labels

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using eta {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = zigzag_order(block_sz)
        self.reverse_order = reverse_zigzag_order(block_sz)

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img)

        # Step 1: Convert RGB to YCbCr
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img_y = 0.299 * R + 0.587 * G + 0.114 * B
        img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)
        cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 2: Split the Y, Cb, and Cr components into BxB blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (h, w) --> (h/B * w/B, B, B)
        cb_blocks = split_into_blocks(cb_downsampled, self.block_sz)  # Cb component, (h/2, w/2) --> (h/2B * w/2B, B, B)
        cr_blocks = split_into_blocks(cr_downsampled, self.block_sz)  # Cr component, (h/2, w/2) --> (h/2B * w/2B, B, B)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (h/B * w/B, B, B)
        dct_cb_blocks = dct_transform(cb_blocks)  # (h/2B * w/2B, B, B)
        dct_cr_blocks = dct_transform(cr_blocks)  # (h/2B * w/2B, B, B)

        # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks pixel region)
        DCT_blocks = []
        for i in range(dct_cr_blocks.shape[0]):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],  # Y
                dct_y_blocks[self.index[i] + 1],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],  # Y
                dct_cb_blocks[i],  # Cb
                dct_cr_blocks[i],  # Cr
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz * self.block_sz)  # (num_tokens, 6, B**2)

        # Step 5: scale into [-1, 1]
        assert DCT_blocks.shape == (self.tokens, 6, self.block_sz * self.block_sz)
        DCT_blocks[:, :4:] = DCT_blocks[:, :4:] / self.Y_bound
        DCT_blocks[:, 4, :] = DCT_blocks[:, 4, :] / self.Y_bound
        DCT_blocks[:, 5, :] = DCT_blocks[:, 5, :] / self.Y_bound

        # Step 6: reorder coe from low to high freq, then mask out high-freq signals
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # organize freqs in the zigzag order
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (num_tokens, 6, B**2) --> (num_tokens, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (num_tokens, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32

        label = np.array(self.labels[idx], dtype=np.int64)

        return DCT_blocks, label


class CelebA(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        """
        manually download dataset: https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&authuser=0
        then do center crop to 64x64 and set the image folder as the following 'path'
        """
        self.train = DCT_4YCbCr(
            path=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/fid_stats_celeba64_all.npz'

    @property
    def has_label(self):
        return False


class FFHQ128(DatasetFactory):
    def __init__(self, path, resolution=128, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            path=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/fid_stats_ffhq128_jpg.npz'

    @property
    def has_label(self):
        return False


class FFHQ256(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            path=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/fid_stats_ffhq256_jpg.npz'

    @property
    def has_label(self):
        return False


class FFHQ512(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            path=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/fid_stats_ffhq512_jpg.npz'

    @property
    def has_label(self):
        return False


class AFHQ512(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            path=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'data/scratch/U-ViT2/assets/fid_stats/fid_stats_afhq512_jpg.npz'

    @property
    def has_label(self):
        return False


class ImageNet64(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = DCT_4YCbCr_cond(
            img_sz=resolution, tokens=tokens, train_files=train_files, labels=train_labels,
            low_freqs=low_freqs, block_sz=block_sz, Y_bound=Y_bound,
        )

        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return f'data/scratch/U-ViT2/assets/fid_stats/fid_stats_imgnet64_jpg.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# MS COCO
def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'ffhq128':
        return FFHQ128(**kwargs)
    elif name == 'ffhq256':
        return FFHQ256(**kwargs)
    elif name == 'ffhq512':
        return FFHQ512(**kwargs)
    elif name == 'afhq512':
        return AFHQ512(**kwargs)
    elif name == 'imgnet64':
        return ImageNet64(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == 'acdc_uncond':
        return ACDCUncond(**kwargs)
    elif name == 'acdc_uncond_images':
        return ACDCUncondImages(**kwargs)
    elif name == 'acdc_cond':
        return ACDCCond(**kwargs)
    elif name == 'echonet':
        return EchoNetUncond(**kwargs)
    elif name == 'echonet_images':
        return EchoNetUncondImages(**kwargs)
    else:
        raise NotImplementedError(name)
