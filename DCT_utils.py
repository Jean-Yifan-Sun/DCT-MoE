import numpy as np
import cv2


def split_into_blocks(image, block_sz):
    blocks = []
    for i in range(0, image.shape[0], block_sz):
        for j in range(0, image.shape[1], block_sz):
            blocks.append(image[i:i + block_sz, j:j + block_sz])  # first row, then column
    return np.array(blocks)

def combine_blocks(blocks, height, width, block_sz):
    image = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            image[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return image

def dct_transform(blocks, shift=True):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) # no shift required for cv2.dct
        if shift:
            dct_block = dct_block - 128  # Shift to range [-128, 127]
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks, shift=True):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        if shift:
            idct_block = idct_block + 128  # Shift back
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)


def zigzag_order(block_sz=8):
    index_list = []

    # Iterate over each diagonal defined by the sum of row and column indices
    for s in range(2 * (block_sz - 1) + 1):
        temp = []  # Initialize a temporary list to collect indices in the current diagonal
        start = max(0, s - block_sz + 1)  # Calculate starting and ending points of the diagonal
        end = min(s, block_sz - 1)

        for i in range(start, end + 1):  # Collect indices in the current diagonal
            temp.append((i, s - i))

        if s % 2 == 0:  # Reverse the diagonal elements if the sum of indices is even
            temp.reverse()

        index_list.extend(temp)  # Convert 2D indices to 1D and append to the main list

    return [i * block_sz + j for i, j in index_list]  # Convert tuple (i, j) to index i * B + j


def reverse_zigzag_order(block_sz=8):
    zigzag_indices = zigzag_order(block_sz)  # Get the zigzag order list
    reverse_order = [0] * (block_sz * block_sz)  # Initialize an array of the same size to store the reverse order

    # Populate the reverse order list where the index is the original position,
    # and the value is the new position according to the zigzag order
    for index, value in enumerate(zigzag_indices):
        reverse_order[value] = index

    return reverse_order

def get_macroblock_indices(image_shape, block_sz, x, y):
    """
    Get the index permutation for macroblock ordering.
    
    Returns:
    - forward_indices: indices to go from block order to macroblock order
    - reverse_indices: indices to go from macroblock order back to block order
    """
    H, W = image_shape
    
    # Basic validation
    if H % block_sz != 0 or W % block_sz != 0:
        raise ValueError(f"Image {H}x{W} not divisible by block size {block_sz}")
    
    blocks_per_row = W // block_sz
    blocks_per_col = H // block_sz
    
    if blocks_per_row % x != 0 or blocks_per_col % y != 0:
        raise ValueError(f"Blocks {blocks_per_col}x{blocks_per_row} not divisible by {y}x{x}")
    
    # Calculate total blocks and macroblocks
    total_blocks = blocks_per_row * blocks_per_col
    macroblocks_per_row = blocks_per_row // x
    macroblocks_per_col = blocks_per_col // y
    # total_macroblocks = macroblocks_per_row * macroblocks_per_col
    
    # Create forward permutation: block index -> macroblock index
    forward_indices = []
    
    # Iterate through macroblocks in row-major order
    for macro_row in range(macroblocks_per_col):
        for macro_col in range(macroblocks_per_row):
            # For each macroblock, get all its block indices in row-major order
            for i in range(y):
                for j in range(x):
                    block_row = macro_row * y + i
                    block_col = macro_col * x + j
                    block_idx = block_row * blocks_per_row + block_col
                    forward_indices.append(block_idx)
    
    # Create reverse permutation
    reverse_indices = [0] * total_blocks
    for new_idx, old_idx in enumerate(forward_indices):
        reverse_indices[old_idx] = new_idx
    
    # metadata = {
    #     'image_shape': image_shape,
    #     'block_sz': block_sz,
    #     'x': x, 'y': y,
    #     'total_blocks': total_blocks
    # }
    
    return forward_indices, reverse_indices

def blocks_to_macro_blocks(blocks, forward_indices):
    """Convert blocks to macroblock order using index permutation"""
    return [blocks[i] for i in forward_indices]

def macro_blocks_to_blocks(macro_blocks, reverse_indices):
    """Convert macroblocks back to original block order"""
    return [macro_blocks[i] for i in reverse_indices]