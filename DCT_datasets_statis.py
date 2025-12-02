from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import shutil
from DCT_utils import split_into_blocks, combine_blocks, dct_transform, idct_transform, zigzag_order, reverse_zigzag_order
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import torch

def convert_png_to_jpg(folder_path):
    cnt = 0
    for subdir, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith(".png"):
                file_path = os.path.join(subdir, file)
                img = Image.open(file_path)
                rgb_img = img.convert('RGB')
                new_filename = file[:-4] + '.jpg'
                new_file_path = os.path.join(subdir, new_filename)
                rgb_img.save(new_file_path, "JPEG")
                os.remove(file_path)  # delete the original PNG file
                cnt += 1

    print(f"{cnt} imgs convertedfrom png to jpg in the folder {folder_path}")


def sample_50k_images(src_folder=None, dest_folder=None, sample_size=50000, has_subfolders=False):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"{dest_folder} created")

    if has_subfolders:
        fname_and_abspath = []
        for sub in os.listdir(src_folder):
            for img in os.listdir(os.path.join(src_folder, sub)):
                fname_and_abspath.append([f"{sub}_{img}", os.path.join(os.path.join(src_folder, sub), img)])
        sampled_files = random.sample(fname_and_abspath, sample_size)

        # Copy the sampled images to the destination folder
        for sample in tqdm(sampled_files):
            f_name = sample[0]
            source_path = sample[1]
            destination_path = os.path.join(dest_folder, f_name)
            shutil.copy2(source_path, destination_path)

    else:
        all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        sampled_files = random.sample(all_files, sample_size)

        # Copy the sampled images to the destination folder
        for file_name in tqdm(sampled_files):
            source_path = os.path.join(src_folder, file_name)
            destination_path = os.path.join(dest_folder, file_name)
            shutil.copy2(source_path, destination_path)


def image_to_DCT_array(dataset=None, img_folder=None, block_sz=8, coe=None, need_batch=False, dest_folder='/home/mang/Downloads'):
    Y_coe = []
    Cb_coe = []
    Cr_coe = []
    file_list = os.listdir(img_folder)
    print(f"found {len(file_list)} images in {img_folder}")
    cnt = 0
    os.makedirs(dest_folder, exist_ok=True)

    for filename in tqdm(file_list):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path).convert('RGB')
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

            # Step 2: Split the Y, Cb, and Cr components into 8x8 blocks
            # Step 3: Apply DCT on each block
            if coe.lower() == 'y':
                y_blocks = split_into_blocks(img_y, block_sz)  # Y component, (64, 64) --> (64, 8, 8)
                dct_y_blocks = dct_transform(y_blocks)  # (64, 8, 8)
                dct_y_blocks = dct_y_blocks.astype(np.float16)
                Y_coe.append(dct_y_blocks.reshape(-1, block_sz*block_sz))
            elif coe.lower() == 'cb':
                cb_blocks = split_into_blocks(cb_downsampled, block_sz)  # Cb component, (32, 32) --> (16, 8, 8)
                dct_cb_blocks = dct_transform(cb_blocks)  # (16, 8, 8)
                dct_cb_blocks = dct_cb_blocks.astype(np.float16)
                Cb_coe.append(dct_cb_blocks.reshape(-1, block_sz*block_sz))
            elif coe.lower() == 'cr':
                cr_blocks = split_into_blocks(cr_downsampled, block_sz)  # Cr component, (32, 32) --> (16, 8, 8)
                dct_cr_blocks = dct_transform(cr_blocks)  # (16, 8, 8)
                dct_cr_blocks = dct_cr_blocks.astype(np.float16)
                Cr_coe.append(dct_cr_blocks.reshape(-1, block_sz*block_sz))

            cnt += 1

        if cnt % 10000 == 0 and coe.lower() == 'y' and need_batch:
            Y_coe = np.array(Y_coe).reshape(-1, block_sz * block_sz)
            print(f"yield Y with shape {Y_coe.shape}")
            np.save(f'{dest_folder}/{dataset}_{block_sz}by{block_sz}_y_{cnt // 10000}', Y_coe)
            print(f"y: {Y_coe.shape} saved")
            Y_coe = []

    if not need_batch:
        if coe.lower() == 'y':
            Y_coe = np.array(Y_coe).reshape(-1, block_sz*block_sz)
            print(f"yield Y with shape {Y_coe.shape}")
            np.save(f'{dest_folder}/{dataset}_{block_sz}by{block_sz}_y', Y_coe)
            print(f"y: {Y_coe.shape} saved")

        if coe.lower() == 'cb':
            Cb_coe = np.array(Cb_coe).reshape(-1, block_sz * block_sz)
            print(f"yield Cb with shape {Cb_coe.shape}")
            np.save(f'{dest_folder}/{dataset}_{block_sz}by{block_sz}_cb', Cb_coe)
            print(f"cb: {Cb_coe.shape} saved")

        elif coe.lower() == 'cr':
            Cr_coe = np.array(Cr_coe).reshape(-1, block_sz * block_sz)
            print(f"yield Cr with shape {Cr_coe.shape}")
            np.save(f'{dest_folder}/{dataset}_{block_sz}by{block_sz}_cr', Cr_coe)
            print(f"cr: {Cr_coe.shape} saved")


def DCT_statis_from_array(array_path=None, block_sz=None, tau=98.25, eta=None):
    """Statistics for DCT coefficients including bounds and entropy.
    Args:
        array_path: Path to the .npy file containing DCT coefficients
        block_sz: Block size for DCT (e.g. 4 for 4x4)
        tau: Percentile threshold for filtering outliers
        eta: Scaling factor for entropy calculation
    """
    coe_array = np.load(array_path)
    print(f"{coe_array.shape} loaded from {array_path}")
    folder_path = os.path.dirname(array_path)
    
    # Create stats file path
    stats_file = os.path.join(folder_path, f'dct_stats_{block_sz}x{block_sz}.md')

    if not os.path.exists(stats_file):
        with open(stats_file, 'w') as f:
            f.write(f"# DCT Statistics for {os.path.basename(array_path)}\n\n")
            f.write(f"- Block size: {block_sz}x{block_sz}\n")
            f.write(f"- Tau: {tau}\n")
            f.write(f"- Array shape: {coe_array.shape}\n\n")
            f.close()
    
    with open(stats_file, 'a') as f:

        low_thresh = 100 - tau
        up_thresh = tau
        order = zigzag_order(block_sz)

        # Calculate statistics without eta
        if not eta:
            DCT_coe_bounds = []
            mean, std = [], []
            _min, _max = [], []
            
            f.write("## Coefficient Statistics\n\n")
            f.write("| Index | Mean | Std | Min | Max | Bound |\n")
            f.write("|-------|------|-----|-----|-----|-------|\n")

            for index in range(block_sz * block_sz):
                data = coe_array[:, index].astype(np.float64)
                lower_bound = np.percentile(data, low_thresh)
                upper_bound = np.percentile(data, up_thresh)
                data = data[(data >= lower_bound) & (data <= upper_bound)]
                
                # Save distribution plot
                # os.makedirs(folder_path + f"/DCT_blocksz_{block_sz}/", exist_ok=True)
                # plot_data_distribution(data, 
                #     title=f"Distribution of DCT coe {index} after filtering outliers",
                #     save_path=folder_path + f"/DCT_blocksz_{block_sz}/DCT_coe_{index}_distribution.png")

                m = np.around(np.mean(data), decimals=3)
                s = np.around(np.std(data), decimals=3)
                mn = np.around(lower_bound, decimals=3)
                mx = np.around(upper_bound, decimals=3)
                
                mean.append(m)
                std.append(s)
                _min.append(mn)
                _max.append(mx)

                bound = max(abs(upper_bound), abs(lower_bound))
                bound = np.around(bound, decimals=3)
                DCT_coe_bounds.append(float(bound))

                f.write(f"| {index} | {m:.3f} | {s:.3f} | {mn:.3f} | {mx:.3f} | {bound:.3f} |\n")

            # Write summary statistics in zigzag order
            f.write("\n## Summary Statistics (Zigzag Order)\n\n")
            f.write(f"- Mean: {np.around(np.array(mean)[order], decimals=3).tolist()}\n")
            f.write(f"- Std: {np.around(np.array(std)[order], decimals=3).tolist()}\n")
            f.write(f"- Min: {np.around(np.array(_min)[order], decimals=3).tolist()}\n")
            f.write(f"- Max: {np.around(np.array(_max)[order], decimals=3).tolist()}\n")
            f.write(f"\nBounds ({up_thresh - low_thresh}% confidence): {DCT_coe_bounds}\n")
            f.write(f"Recommended eta: {DCT_coe_bounds[0]}\n")

        # Calculate entropy if eta is provided
        if eta:
            f.write(f"\n## Entropy Analysis (eta={eta})\n\n")
            f.write("| Index | Entropy |\n")
            f.write("|-------|----------|\n")
            
            entropys = []
            for i in range(block_sz ** 2):
                DCT_coe = coe_array[:, i].astype(np.float64)
                lower_bound = np.percentile(DCT_coe, low_thresh)
                upper_bound = np.percentile(DCT_coe, up_thresh)
                filtered_coe = DCT_coe[(DCT_coe > lower_bound) & (DCT_coe < upper_bound)]
                filtered_coe = filtered_coe / eta

                counts, _ = np.histogram(filtered_coe, bins=100, range=(-1, 1))
                probabilities = counts / np.sum(counts)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
                entropy = np.around(entropy, decimals=3)
                entropys.append(float(entropy))
                
                f.write(f"| {i} | {entropy:.3f} |\n")
            
            f.write(f"\nEntropy values (zigzag order): {np.array(entropys)[order].tolist()}\n")

        print(f"Statistics saved to {stats_file}")


def plot_data_distribution(data, title="Data Distribution", figsize=(12, 8), 
                          save_path=None, style='seaborn-v0_8'):
    """
    Comprehensive function to plot data distribution with multiple visualization types
    
    Parameters:
    -----------
    data : array-like, torch.Tensor, list
        Input data sequence
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str or None
        If provided, save plot to this path
    style : str
        Matplotlib style ('seaborn', 'ggplot', 'classic', etc.)
    """
    
    # Convert to numpy array
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.array(data)
    
    # Flatten if multi-dimensional
    if data_np.ndim > 1:
        data_np = data_np.flatten()
        print(f"Data flattened from {data.shape} to 1D array with {len(data_np)} elements")
    
    # Set style
    # print(plt.style.available)
    plt.style.use(style)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram with KDE
    axes[0, 0].hist(data_np, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    sns.kdeplot(data_np, ax=axes[0, 0], color='red', linewidth=2)
    axes[0, 0].set_title('Histogram with KDE')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    axes[0, 1].boxplot(data_np, vert=True, patch_artist=True)
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Violin plot
    axes[0, 2].violinplot(data_np, showmeans=True, showmedians=True)
    axes[0, 2].set_title('Violin Plot')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Q-Q plot (normality check)
    stats.probplot(data_np, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cumulative distribution
    axes[1, 1].hist(data_np, bins=50, density=True, cumulative=True, 
                   alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Statistical summary
    axes[1, 2].axis('off')
    stats_text = generate_statistical_summary(data_np)
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontfamily='monospace', verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()
    # plt.show()
    
    # return fig

def generate_statistical_summary(data):
    """Generate detailed statistical summary text"""
    stats_dict = {
        'Count': len(data),
        'Mean': np.mean(data),
        'Std': np.std(data),
        'Min': np.min(data),
        '25%': np.percentile(data, 25),
        'Median': np.median(data),
        '75%': np.percentile(data, 75),
        'Max': np.max(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Shapiro-Wilk p': stats.shapiro(data)[1] if len(data) < 5000 else 'N/A (>5000)'
    }
    
    text = "Statistical Summary:\n\n"
    for key, value in stats_dict.items():
        if isinstance(value, float):
            text += f"{key:<15}: {value:>10.4f}\n"
        else:
            text += f"{key:<15}: {value:>10}\n"
    
    return text

def DCT_statis_eta_for_large_arr(array_path=None, block_sz=None, tau=98.25):
    """statistics of eta"""
    DCT_coe_bounds = []
    low_thresh = 100 - tau
    up_thresh = tau

    coe_array_1 = np.load(f"{array_path}_1.npy")[:, 0].astype(np.float64)
    coe_array_2 = np.load(f"{array_path}_2.npy")[:, 0].astype(np.float64)
    coe_array_3 = np.load(f"{array_path}_3.npy")[:, 0].astype(np.float64)
    coe_array_4 = np.load(f"{array_path}_4.npy")[:, 0].astype(np.float64)
    coe_array_5 = np.load(f"{array_path}_5.npy")[:, 0].astype(np.float64)
    data = np.concatenate((coe_array_1, coe_array_2, coe_array_3, coe_array_4, coe_array_5), axis=0)
    print(data.shape)

    lower_bound = np.percentile(data, low_thresh)
    upper_bound = np.percentile(data, up_thresh)
    print(f"({up_thresh - low_thresh}%) coe 0 has upper bound {upper_bound} and lower bound {lower_bound}")

    if np.abs(upper_bound) > np.abs(lower_bound):
        upper_bound = np.around(np.abs(upper_bound), decimals=3)
        DCT_coe_bounds.append(upper_bound)
    else:
        lower_bound = np.around(np.abs(lower_bound), decimals=3)
        DCT_coe_bounds.append(np.abs(lower_bound))

    print(f"{up_thresh - low_thresh} percentile bound is {DCT_coe_bounds}:")
    print(f"eta is {DCT_coe_bounds[0]}")


def DCT_statis_entropy_for_large_arr(array_path=None, block_sz=None, tau=98.25, eta=None):
    """approximate the entropy by histogram"""
    low_thresh = 100 - tau
    up_thresh = tau
    entropys = []
    for i in range(block_sz ** 2):
        # compromise to out-of-memory issue
        coe_array_1 = np.load(f"{array_path}_1.npy")[:, i].astype(np.float64)
        coe_array_2 = np.load(f"{array_path}_2.npy")[:, i].astype(np.float64)
        coe_array_3 = np.load(f"{array_path}_3.npy")[:, i].astype(np.float64)
        coe_array_4 = np.load(f"{array_path}_4.npy")[:, i].astype(np.float64)
        coe_array_5 = np.load(f"{array_path}_5.npy")[:, i].astype(np.float64)
        DCT_coe = np.concatenate((coe_array_1, coe_array_2, coe_array_3, coe_array_4, coe_array_5), axis=0)
        print(f"compute entropy of coe {i} using {DCT_coe.shape}")

        lower_bound = np.percentile(DCT_coe, low_thresh)
        upper_bound = np.percentile(DCT_coe, up_thresh)
        filtered_coe = DCT_coe[(DCT_coe > lower_bound) & (DCT_coe < upper_bound)]
        filtered_coe = filtered_coe / eta  # first get Y_bound, then compute entropy

        counts, bin_edges = np.histogram(filtered_coe, bins=100, range=(-1, 1))
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)
        entropy = np.around(entropy, decimals=3)
        entropys.append(entropy)

    print(f"entropy: {entropys}")

def mask_high_freq_coe_from_img_folder(img_folder=None, save_folder=None, img_sz=256, block_sz=4, low_freqs=None):

    # parameters of DCT transform
    Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
    Y_blocks_per_row = int(img_sz / block_sz)
    cb_blocks_per_row = int((img_sz / block_sz) / 2)
    index = []  # index of Y if merging 2*2 Y-block area
    for row in range(0, Y, int(2 * Y_blocks_per_row)):  # 0, 32, 64...
        for col in range(0, Y_blocks_per_row, 2):  # 0, 2, 4...
            index.append(row + col)
    assert len(index) == int(Y / 4)

    tokens = int((img_sz / (block_sz * 2))**2)
    print(f"num of tokens: {tokens}")
    num_y_blocks = tokens * 4
    num_cb_blocks = tokens

    low2high_order = zigzag_order(block_sz=block_sz)
    reverse_order = reverse_zigzag_order(block_sz=block_sz)

    # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
    cb_index = [i for i in range(4, tokens, 6)]
    cr_index = [i for i in range(5, tokens, 6)]
    y_index = [i for i in range(0, tokens) if i not in cb_index and i not in cr_index]
    assert len(y_index) + len(cb_index) + len(cr_index) == tokens

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"folder {save_folder} created")

    cnt = 0
    for filename in tqdm(os.listdir(img_folder)):
        cnt += 1
        if cnt <= 50000:
            """forward DCT"""
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path).convert('RGB')
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
            y_blocks = split_into_blocks(img_y, block_sz)  # Y component, (64, 64) --> (256, 4, 4)
            cb_blocks = split_into_blocks(cb_downsampled, block_sz)  # Cb component, (32, 32) --> (64, 4, 4)
            cr_blocks = split_into_blocks(cr_downsampled, block_sz)  # Cr component, (32, 32) --> (64, 4, 4)

            # Step 3: Apply DCT on each block
            dct_y_blocks = dct_transform(y_blocks)  # (256, 4, 4)
            dct_cb_blocks = dct_transform(cb_blocks)  # (64, 4, 4)
            dct_cr_blocks = dct_transform(cr_blocks)  # (64, 4, 4)

            # Step 4: reorder coe from low to high freq, then mask out high-freq signals
            dct_y_blocks = dct_y_blocks.reshape(num_y_blocks, block_sz * block_sz)
            dct_cb_blocks = dct_cb_blocks.reshape(num_cb_blocks, block_sz * block_sz)
            dct_cr_blocks = dct_cr_blocks.reshape(num_cb_blocks, block_sz * block_sz)

            dct_y_blocks = dct_y_blocks[:, low2high_order]
            dct_y_blocks = dct_y_blocks[:, :low_freqs]
            dct_cb_blocks = dct_cb_blocks[:, low2high_order]
            dct_cb_blocks = dct_cb_blocks[:, :low_freqs]
            dct_cr_blocks = dct_cr_blocks[:, low2high_order]
            dct_cr_blocks = dct_cr_blocks[:, :low_freqs]


            """Inverse DCT"""
            DCT_Y = np.zeros((num_y_blocks, block_sz * block_sz))
            DCT_Cb = np.zeros((num_cb_blocks, block_sz * block_sz))
            DCT_Cr = np.zeros((num_cb_blocks, block_sz * block_sz))

            DCT_Y[:, :low_freqs] = dct_y_blocks
            DCT_Cb[:, :low_freqs] = dct_cb_blocks
            DCT_Cr[:, :low_freqs] = dct_cr_blocks

            DCT_Y = DCT_Y[:, reverse_order]
            DCT_Cb = DCT_Cb[:, reverse_order]
            DCT_Cr = DCT_Cr[:, reverse_order]

            DCT_Y = DCT_Y.reshape(num_y_blocks, block_sz, block_sz)  # (256, 16) --> (256, 4, 4)
            DCT_Cb = DCT_Cb.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)
            DCT_Cr = DCT_Cr.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)

            # Apply Inverse DCT on each block
            idct_y_blocks = idct_transform(DCT_Y)  # (256, B, B)
            idct_cb_blocks = idct_transform(DCT_Cb)  # (64, B, B)
            idct_cr_blocks = idct_transform(DCT_Cr) # (64, B, B)

            # Combine blocks back into images
            height, width = img_sz, img_sz
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

            # Convert to uint8 and save img
            rgb_reconstructed = np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels
            rgb_reconstructed = Image.fromarray(rgb_reconstructed)
            save_path = os.path.join(save_folder, filename)
            rgb_reconstructed.save(save_path)

        else:
            break

def calculate_block_stats(img_folder=None, block_sz=8):
    """
    Calculates the mean and standard deviation for each position within the blocks
    of the Y channel for all images in a dataset.

    :param img_folder: Path to the folder containing images.
    :param block_sz: The size of the blocks to divide the images into.
    :return: A tuple containing two numpy arrays: (mean, std_dev),
             both of shape (block_sz, block_sz).
    """
    sum_of_values = np.zeros((block_sz, block_sz), dtype=np.float64)
    sum_of_squares = np.zeros((block_sz, block_sz), dtype=np.float64)
    total_blocks = 0

    file_list = os.listdir(img_folder)
    print(f"Found {len(file_list)} images in {img_folder}. Starting statistics calculation...")

    for filename in tqdm(file_list):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)

            # Convert RGB to Y channel
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]
            img_y = 0.299 * R + 0.587 * G + 0.114 * B

            # Split Y channel into blocks
            y_blocks = split_into_blocks(img_y, block_sz)  # Shape: (num_blocks, block_sz, block_sz)
            dct_y_blocks = dct_transform(y_blocks)  # (64, 8, 8)
            y_blocks = dct_y_blocks.astype(np.float64)
            if y_blocks.shape[0] > 0:
                # Accumulate sums for mean and variance calculation
                sum_of_values += np.sum(y_blocks, axis=0)
                sum_of_squares += np.sum(np.square(y_blocks), axis=0)
                total_blocks += y_blocks.shape[0]

    if total_blocks == 0:
        print("No blocks were processed. Please check the image folder and block size.")
        return None, None

    # Calculate mean
    mean = sum_of_values / total_blocks

    # Calculate variance and standard deviation
    # Var(X) = E[X^2] - (E[X])^2
    variance = (sum_of_squares / total_blocks) - np.square(mean)
    std_dev = np.sqrt(variance)

    print(f"\nCalculation finished. Processed a total of {total_blocks} blocks.")
    print("\nMean for each position in the block:")
    print(np.around(mean, decimals=2))
    print("\nStandard Deviation for each position in the block:")
    print(np.around(std_dev, decimals=2))

    return mean, std_dev

def calculate_block_stats_y_channel(img_folder=None, block_sz=8, tau=98.0):
    """
    Calculates the mean and standard deviation for each position within the Y-channel blocks
    across an entire dataset, with outlier filtering and results in zigzag order.

    :param img_folder: Path to the folder containing images.
    :param block_sz: The size of the blocks to divide the images into (e.g., 4 for 4x4).
    :param tau: The percentile to use for filtering extreme values (e.g., 98.0).
                This keeps data between the (100-tau)/2 and (100+tau)/2 percentiles.
    :return: A tuple of (mean_zigzag, std_zigzag).
    """
    # Store all values for each position in the block, e.g., 16 lists for a 4x4 block.
    num_positions = block_sz * block_sz
    all_values = [[] for _ in range(num_positions)]
    total_blocks = 0

    file_list = os.listdir(img_folder)
    print(f"Found {len(file_list)} images. Starting to collect block data...")

    for filename in tqdm(file_list, desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(img_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)

                # Convert RGB to Y channel (Luma component)
                img_y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

                # Split Y channel into blocks
                y_blocks = split_into_blocks(img_y, block_sz)  # Shape: (num_blocks, block_sz, block_sz)
                dct_y_blocks = dct_transform(y_blocks)  # (64, 8, 8)
                y_blocks = dct_y_blocks.astype(np.float64)
                if y_blocks.shape[0] > 0:
                    total_blocks += y_blocks.shape[0]
                    # Reshape to (num_blocks, num_positions) and append to lists
                    reshaped_blocks = y_blocks.reshape(-1, num_positions)
                    for i in range(num_positions):
                        all_values[i].extend(reshaped_blocks[:, i])
            except Exception as e:
                print(f"Warning: Could not process file {filename}. Error: {e}")

    if total_blocks == 0:
        print("No blocks were processed. Please check the image folder and block size.")
        return None, None

    print(f"\nData collection complete. Processed {total_blocks} blocks.")
    print("Calculating statistics with outlier filtering...")

    means = np.zeros(num_positions)
    stds = np.zeros(num_positions)
    low_thresh = (100.0 - tau) / 2.0
    up_thresh = (100.0 + tau) / 2.0

    for i in tqdm(range(num_positions), desc="Calculating stats"):
        position_data = np.array(all_values[i])
        
        # Calculate percentile bounds to filter outliers
        lower_bound = np.percentile(position_data, low_thresh)
        upper_bound = np.percentile(position_data, up_thresh)
        
        # Filter the data
        filtered_data = position_data[(position_data >= lower_bound) & (position_data <= upper_bound)]
        
        if filtered_data.size > 0:
            means[i] = np.mean(filtered_data)
            stds[i] = np.std(filtered_data)
        else:
            # Handle case where filtering removes all data
            means[i] = np.mean(position_data) # Fallback to unfiltered mean
            stds[i] = np.std(position_data)   # Fallback to unfiltered std

    # Get zigzag order and reorder the results
    order = zigzag_order(block_sz)
    mean_zigzag = np.around(means[order], decimals=3)
    std_zigzag = np.around(stds[order], decimals=3)

    print(f"\nStatistics calculated using the central {tau}% of data.")
    print("\nMean for each position (in Zigzag Order):")
    print(mean_zigzag)
    print("\nStandard Deviation for each position (in Zigzag Order):")
    print(std_zigzag)

    return mean_zigzag, std_zigzag


if __name__ == "__main__":

    """cifar10"""
    # # draw 50k samples from the RGB dataset, convert then into DCT arrays for later DCT statis
    # image_to_DCT_array(dataset='cifar10', img_folder='/home/mang/Downloads/cifar10/cifar_train', block_sz=2, coe='y')
    #
    # # get eta
    # DCT_statis_from_array(array_path='/home/mang/Downloads/cifar10_2by2_y.npy', block_sz=2, tau=98.25)
    #
    # # get entropy for Entropy-Based Frequency Reweighting (EBFR)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/cifar10_2by2_y.npy', block_sz=2, tau=98.25, eta=242.382)
    #
    """celeba 64"""
    # sample_50k_images(src_folder='/home/mang/Downloads/celeba/celeba64',
    #                   dest_folder='/home/mang/Downloads/celeba64_50k')
    # image_to_DCT_array(dataset='celeba64', img_folder='/home/mang/Downloads/celeba64_50k', block_sz=2, coe='y')
    # DCT_statis_from_array(array_path='/home/mang/Downloads/celeba64_2by2_y.npy',
    #                       block_sz=2, tau=98.25)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/celeba64_2by2_y.npy',
    #                       block_sz=2, tau=98.25, eta=244.925)
    #
    """imagenet 64"""
    # sample_50k_images(src_folder='/home/mang/Downloads/imagenet64/train',
    #                   dest_folder='/home/mang/Downloads/imagenet64_50k')
    # image_to_DCT_array(dataset='imagenet64', img_folder='/home/mang/Downloads/imagenet64_50k', block_sz=2, coe='y')
    # DCT_statis_from_array(array_path='/home/mang/Downloads/imagenet64_2by2_y.npy',
    #                       block_sz=2, tau=98.25)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/imagenet64_2by2_y.npy',
    #                       block_sz=2, tau=98.25, eta=247.125)
    #
    """ffhq 128"""
    # sample_50k_images(src_folder='/home/mang/Downloads/ffhq128/ffhq128',
    #                   dest_folder='/home/mang/Downloads/ffhq128_50k')
    # image_to_DCT_array(dataset='ffhq128', img_folder='/home/mang/Downloads/ffhq128_50k', block_sz=4, coe='y')
    # DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq128_4by4_y.npy',
    #                       block_sz=4, tau=98.25)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq128_4by4_y.npy',
    #                       block_sz=4, tau=98.25, eta=480.25)

    # """decide m* (Eq. (6) in the paper)"""
    # # get P(dct_data(m)), where m = B*B - low_freqs
    # # then compute FID(P_data, P(dct_data(m))) using repo https://github.com/mseitzer/pytorch-fid
    # mask_high_freq_coe_from_img_folder(img_folder='/home/mang/Downloads/ffhq128_50k',
    #                                    save_folder='/home/mang/Downloads/recon_ffhq128_coe9',
    #                                    img_sz=128, block_sz=4, low_freqs=9)

    #
    """FFHQ 256"""
    # sample_50k_images(src_folder='/home/mang/Downloads/ffhq256/ffhq256',
    #                   dest_folder='/home/mang/Downloads/ffhq256_50k')
    # image_to_DCT_array(dataset='ffhq256', img_folder='/home/mang/Downloads/ffhq256_50k', block_sz=4, coe='y')
    # DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq256_4by4_y.npy',
    #                       block_sz=4, tau=98.25)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq256_4by4_y.npy',
    #                       block_sz=4, tau=98.25, eta=485.0)

    # mask_high_freq_coe_from_img_folder(img_folder='/home/mang/Downloads/ffhq256_50k',
    #                                    save_folder='/home/mang/Downloads/recon_ffhq256_coe8',
    #                                    img_sz=256, block_sz=4, low_freqs=8)

    """FFHQ 512"""
    # sample_50k_images(src_folder='/home/mang/Downloads/ffhq512_jpg/ffhq512',
    #                   dest_folder='/home/mang/Downloads/ffhq512_50k')
    # image_to_DCT_array(dataset='ffhq512', img_folder='/home/mang/Downloads/ffhq512_50k', block_sz=8, coe='y', need_batch=True)
    # DCT_statis_eta_for_large_arr(array_path='/home/mang/Downloads/ffhq512_8by8_y.npy', block_sz=8, tau=98.25)
    # DCT_statis_entropy_for_large_arr(array_path='/home/mang/Downloads/ffhq512_8by8_y.npy', block_sz=8, tau=98.25, eta=969.5)

    # mask_high_freq_coe_from_img_folder(img_folder='/home/mang/Downloads/ffhq512_50k',
    #                                    save_folder='/home/mang/Downloads/recon_ffhq512_coe18',
    #                                    img_sz=512, block_sz=8, low_freqs=18)

    """AFHQv2 512"""
    # convert_png_to_jpg(folder_path='/home/mang/Downloads/afhq512_jpg')  # AFHQv2 contains total 15803 images
    # sample_50k_images(src_folder='/home/mang/Downloads/afhq512_jpg',
    #                   dest_folder='/home/mang/Downloads/afhq512_15k',
    #                   sample_size=15803, has_subfolders=True)  # use the whole dataset for DCT statis
    # image_to_DCT_array(dataset='afhq512', img_folder='/home/mang/Downloads/afhq512_15k', block_sz=8, coe='y',
    #                    need_batch=False)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/afhq512_8by8_y.npy',
    #                       block_sz=8, tau=98.25)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/afhq512_8by8_y.npy',
    #                       block_sz=8, tau=98.25, eta=928.0)

    # mask_high_freq_coe_from_img_folder(img_folder='/home/mang/Downloads/afhq512_15k',
    #                                    save_folder='/home/mang/Downloads/recon_afhq512_coe18',
    #                                    img_sz=512, block_sz=8, low_freqs=18)
    """ACDC 96"""
    # /bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/ACDC/25351_JPGs
    # convert_png_to_jpg(folder_path='/home/mang/Downloads/afhq512_jpg')  # ACDC contains total 25351 images
    # sample_50k_images(src_folder='/home/mang/Downloads/afhq512_jpg',
    #                   dest_folder='/home/mang/Downloads/afhq512_15k',
    #                   sample_size=15803, has_subfolders=True)  # use the whole dataset for DCT statis
    # image_to_DCT_array(dataset='acdc', img_folder='data/scratch/datasets/ACDC/25351_JPGs', block_sz=8, coe='y',
    #                    need_batch=False,dest_folder='data/scratch/datasets/ACDC')
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_8by8_y.npy',
    #                       block_sz=8, tau=98.25)
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_8by8_y.npy',
    #                       block_sz=8, tau=98.25, eta=1000)

    # mask_high_freq_coe_from_img_folder(img_folder='data/scratch/datasets/ACDC/25351_JPGs',
    #                                    save_folder='data/scratch/datasets/ACDC/recon_acdc_coe18',
    #                                    img_sz=96, block_sz=8, low_freqs=32)
    
    # image_to_DCT_array(dataset='acdc', img_folder='data/scratch/datasets/ACDC/JPGs/25351_JPGs', block_sz=2, coe='y',
    #                    need_batch=False,dest_folder='data/scratch/datasets/ACDC')
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_2by2_y.npy',
    #                       block_sz=2, tau=98.25)
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_2by2_y.npy',
    #                       block_sz=2, tau=98.25, eta=252.0)

    # mask_high_freq_coe_from_img_folder(img_folder='data/scratch/datasets/ACDC/JPGs/25351_JPGs',
    #                                    save_folder='data/scratch/datasets/ACDC/recon_acdc_coe4',
    #                                    img_sz=96, block_sz=2, low_freqs=4)
    block_sz = 4
    eta = 512.0
    # image_to_DCT_array(dataset='acdc', 
    #                    img_folder='data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs', 
    #                    block_sz=block_sz, 
    #                    coe='y',
    #                    need_batch=False,dest_folder='data/scratch/datasets/ACDC')
    # DCT_statis_from_array(array_path=f'data/scratch/datasets/ACDC/acdc_{block_sz}by{block_sz}_y.npy',
    #                       block_sz=block_sz, 
    #                       tau=98.25)
    # DCT_statis_from_array(array_path=f'data/scratch/datasets/ACDC/acdc_{block_sz}by{block_sz}_y.npy',
    #                       block_sz=block_sz, tau=98.25, eta=eta)
    # calculate_block_stats(img_folder='data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs', block_sz=block_sz)
    # calculate_block_stats_y_channel(img_folder='data/scratch/datasets/ACDC/Unlabeled/Wholeheart/25022_JPGs', block_sz=block_sz, tau=96.0)

    # image_to_DCT_array(dataset='echo', 
    #                    img_folder='/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/EchoNet-Dynamic/images', 
    #                    block_sz=block_sz, 
    #                    coe='y',
    #                    need_batch=False, dest_folder='data/scratch/datasets/EchoNet-Dynamic')
    DCT_statis_from_array(array_path=f'data/scratch/datasets/EchoNet-Dynamic/echo_{block_sz}by{block_sz}_y.npy',
                          block_sz=block_sz, 
                          tau=98.25)
    DCT_statis_from_array(array_path=f'data/scratch/datasets/EchoNet-Dynamic/echo_{block_sz}by{block_sz}_y.npy',
                          block_sz=block_sz, tau=98.25, eta=eta)
    calculate_block_stats(img_folder='/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/EchoNet-Dynamic/images', block_sz=block_sz)
    calculate_block_stats_y_channel(img_folder='/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/datasets/EchoNet-Dynamic/images', block_sz=block_sz, tau=96.0)


    # mask_high_freq_coe_from_img_folder(img_folder='data/scratch/datasets/ACDC/JPGs/25351_JPGs',
    #                                    save_folder='data/scratch/datasets/ACDC/recon_acdc_coe4',
    #                                    img_sz=96, block_sz=4, low_freqs=4)

    # image_to_DCT_array(dataset='acdc_cond', img_folder='data/scratch/datasets/ACDC/Image', block_sz=4, coe='y',
    #                    need_batch=False,dest_folder='data/scratch/datasets/ACDC')
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_cond_4by4_y.npy',
    #                       block_sz=4, tau=98.25)
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_cond_4by4_y.npy',
    #                       block_sz=4, tau=98.25, eta=502.25)

    # mask_high_freq_coe_from_img_folder(img_folder='data/scratch/datasets/ACDC/Image',
    #                                    save_folder='data/scratch/datasets/ACDC/recon_acdc_cond_coe16',
    #                                    img_sz=96, block_sz=4, low_freqs=16)
    
    # image_to_DCT_array(dataset='acdc_cond_label', img_folder='data/scratch/datasets/ACDC/Label_255', block_sz=4, coe='y',
    #                    need_batch=False,dest_folder='data/scratch/datasets/ACDC')
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_cond_label_4by4_y.npy',
    #                       block_sz=4, tau=98.25)
    # DCT_statis_from_array(array_path='data/scratch/datasets/ACDC/acdc_cond_label_4by4_y.npy',
    #                       block_sz=4, tau=98.25, eta=512)

    # mask_high_freq_coe_from_img_folder(img_folder='data/scratch/datasets/ACDC/Label_255',
    #                                    save_folder='data/scratch/datasets/ACDC/recon_acdc_cond_label_coe16',
    #                                    img_sz=96, block_sz=4, low_freqs=16)