"""
Batch script to calculate and compare GFLOPs across multiple models.

Usage:
    python compare_gflops.py --configs configs/acdc*.py
    python compare_gflops.py --model-list model_list.txt
    python compare_gflops.py --config-dir configs --pattern "*greyscale*"
"""

import argparse
import torch
import pandas as pd
import glob
import os
import sys
from pathlib import Path
from datetime import datetime
import json

from calculate_gflops import load_config, get_input_shape, calculate_gflops, calculate_vae_gflops
from utils import get_nnet


def process_models(config_paths, device='cuda', output_csv=None, output_json=None):
    """
    Process multiple model configs and calculate GFLOPs.
    
    Args:
        config_paths: List of config file paths
        device: Device to use ('cuda' or 'cpu')
        output_csv: Optional CSV file to save results
        output_json: Optional JSON file to save results
    
    Returns:
        DataFrame with results
    """
    results = []
    
    for config_path in config_paths:
        if not os.path.exists(config_path):
            print(f"⚠️  Skipping non-existent file: {config_path}")
            continue
        
        try:
            print(f"Processing: {config_path}")
            
            # Load config
            config = load_config(config_path)
            
            # Get model name
            model_name = config.nnet.get('name', 'unknown')
            config_name = Path(config_path).stem
            
            # Initialize model
            model = get_nnet(**config.nnet)
            
            # Get input shape
            input_shape = get_input_shape(config, device)
            
            # Calculate GFLOPs
            result_dict = calculate_gflops(model, input_shape, device=device)
            diffusion_gflops = result_dict['gflops']  # Save original diffusion model GFLOPs
            
            # Check if this is an LDM (Latent Diffusion Model) with VAE
            is_ldm = hasattr(config, 'autoencoder') or 'autoencoder' in config
            if is_ldm:
                # Calculate VAE GFLOPs and add to total
                vae_gflops = calculate_vae_gflops(config, device=device)
                if vae_gflops is not None:
                    # Combine diffusion model GFLOPs with VAE GFLOPs
                    total_gflops = diffusion_gflops + vae_gflops['total_vae_gflops']
                    result_dict['gflops'] = total_gflops
                    result_dict['gflops_breakdown'] = {
                        'diffusion_model': diffusion_gflops,
                        'vae_encoder': vae_gflops['vae_encoder_gflops'],
                        'vae_decoder': vae_gflops['vae_decoder_gflops'],
                        'total': total_gflops
                    }
            
            # Get additional info from config
            depth = config.nnet.get('depth', 'N/A')
            embed_dim = config.nnet.get('embed_dim', 'N/A')
            num_heads = config.nnet.get('num_heads', 'N/A')
            resolution = config.dataset.get('resolution', 'N/A')
            batch_size = config.train.get('batch_size', 'N/A')
            
            # Compile result
            result = {
                'Config File': config_name,
                'Model Type': model_name,
                'GFLOPs': round(result_dict['gflops'], 2),
                'Parameters (M)': round(result_dict['m_params'], 2),
                'Depth': depth,
                'Embed Dim': embed_dim,
                'Num Heads': num_heads,
                'Resolution': resolution,
                'Train Batch Size': batch_size,
            }
            
            results.append(result)
            if 'gflops_breakdown' in result_dict:
                bd = result_dict['gflops_breakdown']
                print(f"  ✓ GFLOPs (Total): {result_dict['gflops']:.2f}")
                print(f"    - Diffusion: {bd['diffusion_model']:.2f}, VAE Encoder: {bd['vae_encoder']:.2f}, VAE Decoder: {bd['vae_decoder']:.2f}")
                print(f"    Params: {result_dict['m_params']:.2f}M")
            else:
                print(f"  ✓ GFLOPs: {result_dict['gflops']:.2f}, Params: {result_dict['m_params']:.2f}M")
            
        except Exception as e:
            print(f"  ✗ Error processing {config_path}: {e}")
            results.append({
                'Config File': Path(config_path).stem,
                'Model Type': 'ERROR',
                'GFLOPs': None,
                'Parameters (M)': None,
                'Depth': 'ERROR',
                'Embed Dim': 'ERROR',
                'Num Heads': 'ERROR',
                'Resolution': 'ERROR',
                'Train Batch Size': 'ERROR',
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by GFLOPs in descending order
    df = df.sort_values('GFLOPs', ascending=False, na_position='last')
    
    # Save results if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to CSV: {output_csv}")
    
    if output_json:
        # Convert to JSON-serializable format
        df_json = df.to_dict('records')
        with open(output_json, 'w') as f:
            json.dump(df_json, f, indent=2)
        print(f"✓ Results saved to JSON: {output_json}")
    
    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*100)
    print("GFLOPS COMPARISON SUMMARY")
    print("="*100)
    
    # Remove rows with errors
    df_valid = df[df['Model Type'] != 'ERROR'].copy()
    
    if len(df_valid) > 0:
        print(f"\nTotal models analyzed: {len(df_valid)}")
        print(f"Max GFLOPs: {df_valid['GFLOPs'].max():.2f}")
        print(f"Min GFLOPs: {df_valid['GFLOPs'].min():.2f}")
        print(f"Mean GFLOPs: {df_valid['GFLOPs'].mean():.2f}")
        print(f"Median GFLOPs: {df_valid['GFLOPs'].median():.2f}")
        
        print(f"\nMax Parameters: {df_valid['Parameters (M)'].max():.2f}M")
        print(f"Min Parameters: {df_valid['Parameters (M)'].min():.2f}M")
        print(f"Mean Parameters: {df_valid['Parameters (M)'].mean():.2f}M")
    
    print("\n" + "-"*100)
    print("DETAILED RESULTS")
    print("-"*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare GFLOPs across multiple models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all configs matching pattern
  python compare_gflops.py --config-dir configs --pattern "*greyscale_moe*"
  
  # Compare specific configs
  python compare_gflops.py --configs configs/acdc*.py
  
  # Load from file list
  python compare_gflops.py --model-list my_models.txt
  
  # Save results to files
  python compare_gflops.py --config-dir configs --pattern "*" --output-csv results.csv --output-json results.json
        """
    )
    
    # Config selection options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--configs', type=str, nargs='+',
                              help='Space-separated list of config files or glob patterns')
    config_group.add_argument('--config-dir', type=str,
                              help='Directory containing config files')
    config_group.add_argument('--model-list', type=str,
                              help='Text file with one config path per line')
    
    # Other options
    parser.add_argument('--pattern', type=str, default='*.py',
                        help='Pattern for config files when using --config-dir (default: *.py)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--output-csv', type=str,
                        help='Save results to CSV file')
    parser.add_argument('--output-json', type=str,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Determine config paths
    config_paths = []
    
    if args.configs:
        # Handle glob patterns
        for pattern in args.configs:
            paths = glob.glob(pattern)
            config_paths.extend(paths)
    
    elif args.config_dir:
        # Get all files matching pattern in directory
        pattern = os.path.join(args.config_dir, args.pattern)
        config_paths = sorted(glob.glob(pattern))
    
    elif args.model_list:
        # Read from file
        with open(args.model_list, 'r') as f:
            config_paths = [line.strip() for line in f if line.strip()]
    
    if not config_paths:
        print("❌ No config files found!")
        sys.exit(1)
    
    print(f"Found {len(config_paths)} config files to process")
    print()
    
    # Process models
    df = process_models(
        config_paths,
        device=args.device,
        output_csv=args.output_csv,
        output_json=args.output_json
    )
    
    # Print summary
    print_summary(df)


if __name__ == '__main__':
    main()
