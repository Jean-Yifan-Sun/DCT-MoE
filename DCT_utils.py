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

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) # no shift required for cv2.dct
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        # idct_block = idct_block + 128  # Shift back
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

def reorder_to_squares(blocks, num_blocks_y, num_blocks_x):
    """
    Reorders blocks from row-major to a "square-shell" order.
    
    The first r^2 blocks in the new list will form the r x r top-left
    square of blocks from the original image.
    
    Args:
        blocks (np.array): The input blocks in row-major order.
        num_blocks_y (int): The number of blocks in the y-direction (rows).
        num_blocks_x (int): The number of blocks in the x-direction (cols).

    Returns:
        tuple:
            - new_blocks (np.array): The reordered blocks.
            - forward_permutation (np.array): The permutation array used.
                                              (new_index -> original_index)
    """
    
    # This array will store the original (row-major) indices 
    # in the new "square-shell" order.
    forward_permutation = []
    
    # We iterate up to the largest dimension to form squares
    max_dim = max(num_blocks_y, num_blocks_x)
    
    for r in range(1, max_dim + 1):
        # r is the side length of the current square (e.g., 1, 2, 3...)
        # We are adding the "L-shaped" shell for square r.
        
        # 1. Add the new right column of the shell (from top to bottom)
        j = r - 1  # Column index of the new shell
        if j < num_blocks_x:
            for i in range(r): # Row index from 0 to r-1
                if i < num_blocks_y:
                    # Convert 2D block index (i, j) to 1D row-major index
                    original_flat_index = i * num_blocks_x + j
                    forward_permutation.append(original_flat_index)
                    
        # 2. Add the new bottom row of the shell (from left to right)
        #    (We skip the corner, as it was added in the column part)
        i = r - 1 # Row index of the new shell
        if i < num_blocks_y:
            for j in range(r - 1): # Col index from 0 to r-2 (skips corner)
                if j < num_blocks_x:
                    original_flat_index = i * num_blocks_x + j
                    forward_permutation.append(original_flat_index)

    # Convert to a NumPy array for indexing
    forward_permutation = np.array(forward_permutation)
    
    # Use the permutation array to reorder the blocks
    # This is called "fancy indexing"
    new_blocks = blocks[forward_permutation]
    
    return new_blocks, forward_permutation

def restore_original_order(new_blocks, forward_permutation):
    """
    Restores the original row-major block order from the "square-shell" order.
    
    Args:
        new_blocks (np.array): The reordered blocks.
        forward_permutation (np.array): The permutation array from the
                                        reorder_to_squares function.
    
    Returns:
        np.array: The blocks in their original row-major order.
    """
    
    # We need the inverse permutation, which maps:
    # original_index -> new_index
    # np.argsort() on the forward_permutation gives us exactly this.
    inverse_permutation = np.argsort(forward_permutation)
    
    # Use the inverse permutation to "un-shuffle" the new_blocks array
    # back to its original order.
    original_blocks = new_blocks[inverse_permutation]
    
    return original_blocks

# if __name__ == "__main__":
#     print(zigzag_order(block_sz=8))
#     print(reverse_zigzag_order(block_sz=8))