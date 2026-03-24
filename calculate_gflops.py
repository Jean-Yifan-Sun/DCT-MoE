"""
Script to calculate GFLOPs for different models using ptflops.

Usage:
    python calculate_gflops.py --config configs/model_config.py
    python calculate_gflops.py --config configs/model_config.py --batch_size 1 --device cpu
"""

import argparse
import torch
import ml_collections
import os
import sys
import inspect
from pathlib import Path
from ptflops import get_model_complexity_info
import importlib.util

from utils import get_nnet


def load_config(config_path):
    """Load config from a Python file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the Python config file dynamically
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, 'get_config'):
        return config_module.get_config()
    else:
        raise ValueError(f"Config file {config_path} must have a get_config() function")


def get_input_shape(config, device='cuda'):
    """
    Determine the input shape based on config.
    
    Args:
        config: Configuration dict
        device: Device to use
    
    Returns:
        A tuple (B, C, H, W) for image input, or (B, seq_len) for token input
    """
    # Get model type from config
    nnet_config = config.get('nnet', {})
    model_name = nnet_config.get('name', 'uvit')
    
    # Get dataset config for resolution info
    dataset_config = config.get('dataset', {})
    resolution = dataset_config.get('resolution', 256)
    
    # Default batch size for FLOP calculation
    batch_size = 1
    
    # For greyscale/DCT models, input is flattened DCT coefficients
    if 'greyscale' in model_name or 'DCT' in model_name:
        # These models use DCT coefficients as input
        # The input shape is (batch, DCT_coes * 4)
        low_freqs = nnet_config.get('low_freqs', 0)
        if low_freqs > 0:
            input_dim = low_freqs * 4
        else:
            # Default: use resolution as a proxy
            input_dim = resolution  # This will be adjusted based on actual DCT coes
        return (batch_size, input_dim)
    else:
        # Regular models work with RGB images (3-channel)
        channels = 3
        height = resolution
        width = resolution
        return (batch_size, channels, height, width)


def model_requires_timesteps(model):
    """
    Check if the model requires timesteps argument in forward pass.
    
    Args:
        model: PyTorch model
    
    Returns:
        True if model requires timesteps, False otherwise
    """
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        # Check if 'timesteps' is a required parameter (not optional)
        if 'timesteps' in params:
            param = params['timesteps']
            # It's required if it has no default value
            return param.default == inspect.Parameter.empty
    except Exception:
        pass
    return False


def calculate_gflops(model, input_shape, device='cuda'):
    """
    Calculate GFLOPs using ptflops or manual estimation.
    
    Args:
        model: PyTorch model
        input_shape: Tuple (B, C, H, W)
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        dict with 'flops' and 'params' keys
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    
    # Check if model requires timesteps
    has_timesteps = model_requires_timesteps(model)
    
    # If model requires timesteps, skip ptflops and use manual estimation
    if has_timesteps:
        print(f"  Model requires timesteps, using manual FLOP estimation...")
        return estimate_flops_manual(model, input_shape, device, has_timesteps=True)
    
    # Try ptflops for models without timesteps
    try:
        # Get model complexity info
        flops, params = get_model_complexity_info(
            model,
            input_shape[1:],  # ptflops expects input shape without batch dimension
            as_strings=False,
            print_per_layer_stat=False,
            ignore_modules=[]
        )
        
        return {
            'flops': flops,
            'params': params,
            'gflops': flops / 1e9,
            'm_params': params / 1e6
        }
    except Exception as e:
        print(f"  Warning: ptflops calculation failed with error: {e}")
        print("  Attempting manual FLOP calculation...")
        return estimate_flops_manual(model, input_shape, device, has_timesteps=False)


def estimate_flops_manual(model, input_shape, device='cuda', has_timesteps=False):
    """
    Manual FLOP estimation by counting operations in each layer.
    This is a fallback method when ptflops fails.
    
    Args:
        model: PyTorch model
        input_shape: Tuple (B, feature_dim) or (B, C, H, W)
        device: Device to use
        has_timesteps: Whether the model requires timesteps argument
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    
    total_flops = 0
    total_params = sum(p.numel() for p in model.parameters())
    
    def count_linear_flops(module, input, output):
        """Count FLOPs for linear layers."""
        nonlocal total_flops
        if isinstance(module, torch.nn.Linear):
            batch_size = input[0].shape[0]
            in_features = module.in_features
            out_features = module.out_features
            flops = batch_size * in_features * out_features
            total_flops += flops
    
    def count_conv_flops(module, input, output):
        """Count FLOPs for convolutional layers."""
        nonlocal total_flops
        if isinstance(module, torch.nn.Conv2d):
            batch_size = input[0].shape[0]
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_h, kernel_w = module.kernel_size
            out_h, out_w = output[0].shape[-2:]
            flops = batch_size * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
            if module.bias is not None:
                flops += batch_size * out_channels * out_h * out_w
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(count_linear_flops))
        elif isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv_flops))
    
    # Try to run inference with proper dummy inputs
    forward_failed = False
    with torch.no_grad():
        batch_size = input_shape[0]
        
        try:
            # Create appropriate dummy input based on shape
            if len(input_shape) == 2:
                # Token/DCT input: (batch, feature_dim)
                dummy_input = torch.randn(*input_shape, device=device_obj)
            else:
                # Image input: (batch, C, H, W)
                dummy_input = torch.randn(*input_shape, device=device_obj)
            
            if has_timesteps:
                # Create dummy timesteps with shape (batch_size,)
                dummy_timesteps = torch.randint(0, 1000, (batch_size,), device=device_obj)
                
                # Check if model has optional condition input (y parameter)
                sig = inspect.signature(model.forward)
                if 'y' in sig.parameters:
                    param = sig.parameters['y']
                    # Check if y is optional or required
                    if param.default != inspect.Parameter.empty:
                        # y is optional, so we can pass None or not specify it
                        model(dummy_input, dummy_timesteps)
                    else:
                        # y is required, need to create dummy condition
                        dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device_obj)
                        model(dummy_input, dummy_timesteps, dummy_y)
                else:
                    model(dummy_input, dummy_timesteps)
            else:
                model(dummy_input)
        except Exception as e:
            forward_failed = True
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # If forward pass failed, estimate using Transformer FLOPs formula
    if forward_failed or total_flops == 0:
        total_flops = estimate_transformer_flops(model)
    
    return {
        'flops': total_flops,
        'params': total_params,
        'gflops': total_flops / 1e9,
        'm_params': total_params / 1e6
    }


def estimate_transformer_flops(model):
    """
    Estimate FLOPs for Transformer-based models using standard formula.
    
    Formula for one pass of Transformer block:
    FLOPs = batch_size * seq_len * (12 * embed_dim^2 + 13 * embed_dim * ffn_dim)
    
    Where ffn_dim = mlp_ratio * embed_dim
    
    Args:
        model: PyTorch Transformer model
    
    Returns:
        Estimated FLOPs as integer
    """
    total_flops = 0
    
    # Try to extract Transformer parameters from model
    embed_dim = None
    depth = None
    mlp_ratio = 4
    seq_len = 1024  # Default estimate
    batch_size = 1
    
    # Look for common Transformer attributes
    for name, module in model.named_modules():
        if hasattr(module, 'embed_dim'):
            embed_dim = module.embed_dim
            break
    
    # Try to find depth
    if hasattr(model, 'depth'):
        depth = model.depth
    elif hasattr(model, 'num_layers'):
        depth = model.num_layers
    else:
        # Count transformer blocks
        block_count = 0
        for name, module in model.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                block_count = max(block_count, int(''.join(filter(str.isdigit, name.split('.')[-1]))) if any(c.isdigit() for c in name.split('.')[-1]) else 0)
        depth = max(block_count + 1, 12)  # Default to 12 if not found
    
    # Try to find mlp_ratio
    if hasattr(model, 'mlp_ratio'):
        mlp_ratio = model.mlp_ratio
    
    # Try to estimate sequence length from model
    if hasattr(model, 'tokens'):
        seq_len = model.tokens + 2  # +2 for class and time tokens
    elif hasattr(model, 'num_patches'):
        seq_len = model.num_patches + 2
    else:
        seq_len = 1024
    
    if embed_dim is None:
        embed_dim = 768  # Default assumption
    
    ffn_dim = mlp_ratio * embed_dim
    
    # Standard Transformer FLOPs formula
    # Per block: FFN = 2 * N * d * (4*d) = 8 * N * d^2
    # Per block: Attention = 2 * N * d^2 + 2 * N^2 * d
    # Simplified: FLOPs ≈ 2 * seq_len * embed_dim^2 * depth (per attention head)
    # For full model with mlp: FLOPs ≈ seq_len * depth * (12*embed_dim^2 + 13*embed_dim*ffn_dim)
    
    total_flops = int(seq_len * depth * (12 * embed_dim**2 + 13 * embed_dim * ffn_dim) * batch_size)
    
    return total_flops


def calculate_vae_gflops(config, device='cuda'):
    """
    Calculate GFLOPs for VAE encoder and decoder.
    
    Args:
        config: Configuration dict
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        dict with 'vae_encoder_gflops', 'vae_decoder_gflops', and 'total_vae_gflops'
    """
    try:
        from libs.autoencoder import get_model_diffusers
    except ImportError:
        print("Warning: Could not import VAE from libs.vae")
        return None
    
    device_obj = torch.device(device)
    
    # Load VAE model
    vae_config = config.get('vae', {})
    if not vae_config:
        print("Warning: No VAE config found")
        return None
    
    try:
        vae = get_model_diffusers(config.autoencoder.pretrained_path).to(device_obj)
        vae.eval()
    except Exception as e:
        print(f"Warning: Could not initialize VAE: {e}")
        return None
    
    # Get image resolution
    resolution = config.get('dataset', {}).get('resolution', 256)
    
    # Determine channels (greyscale vs RGB)
    nnet_config = config.get('nnet', {})
    model_name = nnet_config.get('name', '')
    if 'greyscale' in model_name or 'DCT' in model_name:
        channels = 1
    else:
        channels = 3
    
    results = {}
    
    # Calculate encoder GFLOPs
    # Input: original image (B, C, H, W)
    encoder_input_shape = (1, channels, resolution, resolution)
    try:
        encoder_flops, encoder_params = get_model_complexity_info(
            vae.encoder,
            encoder_input_shape[1:],
            as_strings=False,
            print_per_layer_stat=False
        )
        results['vae_encoder_gflops'] = encoder_flops / 1e9
        results['vae_encoder_params'] = encoder_params / 1e6
    except Exception as e:
        print(f"Warning: VAE encoder FLOP calculation failed: {e}")
        results['vae_encoder_gflops'] = 0
        results['vae_encoder_params'] = 0
    
    # Calculate decoder GFLOPs
    # Input: latent representation (B, C, H/8, W/8)
    # Latent scaling depends on your VAE's downsampling factor (usually 8x)
    latent_channels = vae_config.get('latent_channels', 4)
    latent_size = resolution // 8  # Typically 8x downsampling
    decoder_input_shape = (1, latent_channels, latent_size, latent_size)
    
    try:
        decoder_flops, decoder_params = get_model_complexity_info(
            vae.decoder,
            decoder_input_shape[1:],
            as_strings=False,
            print_per_layer_stat=False
        )
        results['vae_decoder_gflops'] = decoder_flops / 1e9
        results['vae_decoder_params'] = decoder_params / 1e6
    except Exception as e:
        print(f"Warning: VAE decoder FLOP calculation failed: {e}")
        results['vae_decoder_gflops'] = 0
        results['vae_decoder_params'] = 0
    
    results['total_vae_gflops'] = results['vae_encoder_gflops'] + results['vae_decoder_gflops']
    
    return results


def print_results(model_name, config_path, diffusion_results, vae_results, input_shape):
    """Pretty print the results including VAE GFLOPs if available."""
    print("\n" + "="*70)
    print(f"Model: {model_name}")
    print(f"Config: {config_path}")
    print("="*70)
    print(f"Input shape (B, C, H, W): {input_shape}")
    
    print("\nDiffusion Model:")
    print(f"  GFLOPs: {diffusion_results['gflops']:.2f}")
    print(f"  Parameters: {diffusion_results['m_params']:.2f}M")
    print(f"  FLOPs: {diffusion_results['flops']:.2e}")
    
    if vae_results:
        print("\nVAE:")
        print(f"  Encoder GFLOPs: {vae_results['vae_encoder_gflops']:.2f}")
        print(f"  Decoder GFLOPs: {vae_results['vae_decoder_gflops']:.2f}")
        print(f"  Total VAE GFLOPs: {vae_results['total_vae_gflops']:.2f}")
        vae_params = vae_results.get('vae_encoder_params', 0) + vae_results.get('vae_decoder_params', 0)
        print(f"  VAE Parameters: {vae_params:.2f}M")
        
        total_gflops = diffusion_results['gflops'] + vae_results['total_vae_gflops']
        total_params = diffusion_results['m_params'] + vae_params
        print(f"\nTotal (Diffusion + VAE):")
        print(f"  GFLOPs: {total_gflops:.2f}")
        print(f"  Parameters: {total_params:.2f}M")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Calculate GFLOPs for models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/model_config.py)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for FLOP calculation (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-layer statistics')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Get model name
    model_name = config.nnet.get('name', 'unknown')
    
    # Initialize model
    print(f"Initializing model: {model_name}")
    model = get_nnet(**config.nnet)
    
    # Determine input shape
    input_shape = get_input_shape(config, args.device)
    print(f"Input shape: {input_shape}")
    
    # Calculate diffusion model GFLOPs
    print("Calculating diffusion model GFLOPs...")
    diffusion_results = calculate_gflops(model, input_shape, device=args.device)
    
    # Calculate VAE GFLOPs if VAE is configured
    vae_results = None
    if config.get('use_vae', True):  # Check if VAE is configured
        print("Calculating VAE GFLOPs...")
        try:
            vae_results = calculate_vae_gflops(config, device=args.device)
        except Exception as e:
            print(f"Warning: Could not calculate VAE GFLOPs: {e}")
    
    # Print results
    print_results(model_name, args.config, diffusion_results, vae_results, input_shape)
    
    # Optional: verify with detailed per-layer stats if verbose
    if args.verbose:
        print("Per-layer analysis (Diffusion Model):")
        print("-" * 70)
        total_params = sum(p.numel() for p in model.named_parameters())
        for name, param in model.named_parameters():
            if len(param.shape) > 1:  # Only show non-bias parameters
                print(f"{name}: {param.numel():,} params")
        print("-" * 70)


if __name__ == '__main__':
    main()
