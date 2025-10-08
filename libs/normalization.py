import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
from absl import logging
import numpy as np

class ReversibleFrequencyAdaptiveNorm(nn.Module):
    """
    Reversible frequency-adaptive normalization with exact inverse
    Based on: "Glow: Generative Flow with Invertible 1x1 Convolutions" (Kingma & Dhariwal, 2018)
    """
    
    def __init__(self, num_freq_bins, eps=1e-5, use_running_stats=False):
        super().__init__()
        """
        num_freq_bins: Number of frequency bins in the input features
        eps: Small constant for numerical stability
        """
        self.num_freq_bins = num_freq_bins
        self.eps = eps
        self.use_running_stats = use_running_stats
        
        # Learnable affine parameters (must be invertible)
        self.log_gamma = nn.Parameter(torch.zeros(num_freq_bins))  # Use log for stability
        self.beta = nn.Parameter(torch.zeros(num_freq_bins))
        
        # Fixed frequency weights (non-learnable for reversibility)
        with torch.no_grad():
            freq_indices = torch.arange(num_freq_bins, dtype=torch.float32)
            self.register_buffer('freq_weights', 1.0 / torch.sqrt(freq_indices + 1.0))

        # Internal cache storage
        self.register_buffer('_cache_mean', None)
        self.register_buffer('_cache_std', None)
        self.register_buffer('_initialized', torch.tensor(False))
        
        # Optional running statistics (like BatchNorm)
        if use_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_freq_bins))
            self.register_buffer('running_var', torch.ones(num_freq_bins))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.momentum = 0.1
        
    def forward(self, x, reverse=False):
        """
        x: [batch_size, seq_len, num_freq_bins]
        reverse: False for normalization, True for denormalization
        cache: Optional cache for storing forward pass statistics
        """
        if not reverse:
            return self._normalize_forward(x)
        else:
            return self._normalize_reverse(x)
    
    def _normalize_forward(self, x, cache=None):
        batch_size, seq_len, num_bins = x.shape
        
        if self.use_running_stats and self.training:
            # Update running statistics during training
            self._update_running_stats(x)

        # Compute statistics (store for reverse pass)
        if self.use_running_stats and not self.training:
            # Use running statistics during inference
            mean = self.running_mean.view(1, 1, -1)
            std = torch.sqrt(self.running_var + self.eps).view(1, 1, -1)
        else:
            # Use batch statistics
            mean = x.mean(dim=[0, 1], keepdim=True)  # [1, 1, num_bins]
            std = x.std(dim=[0, 1], keepdim=True) + self.eps  # [1, 1, num_bins]
        
        # Store statistics in internal cache for reverse pass
        self._cache_mean = mean.detach().clone()
        self._cache_std = std.detach().clone()
        self._initialized = torch.tensor(True)
        
        # Apply normalization with affine transformation
        gamma = torch.exp(self.log_gamma).view(1, 1, -1)  # [1, 1, num_bins]
        beta = self.beta.view(1, 1, -1)  # [1, 1, num_bins]
        weights = self.freq_weights.view(1, 1, -1)  # [1, 1, num_bins]
        
        # Reversible transformation: y = (x - μ) / σ * (γ ⊙ w) + β
        x_normalized = (x - mean) / std
        x_scaled = x_normalized * gamma * weights + beta
        
        # Log determinant for flow-based models (optional)
        # log_det = torch.sum(self.log_gamma) + torch.sum(torch.log(weights.squeeze()))
        # log_det -= torch.sum(torch.log(std.squeeze()))
        
        return x_scaled
    
    def _normalize_reverse(self, y, cache=None):
        """Exact inverse transformation"""
        if not self._initialized:
            raise ValueError("Cache not initialized. Call forward pass first.")
        
        if self._cache_mean is None or self._cache_std is None:
            raise ValueError("Cache is empty. Call forward pass before reverse.")
        
        # Retrieve parameters
        gamma = torch.exp(self.log_gamma).view(1, 1, -1)
        beta = self.beta.view(1, 1, -1)
        weights = self.freq_weights.view(1, 1, -1)
        
        # Reverse transformation: x = ((y - β) / (γ ⊙ w)) * σ + μ
        x_normalized = (y - beta) / (gamma * weights)
        x_original = x_normalized * self._cache_std + self._cache_mean
        
        return x_original
    
    def _update_running_stats(self, x):
        """Update running statistics like BatchNorm"""
        if self.num_batches_tracked is not None:
            self.num_batches_tracked += 1
        
        # Compute batch statistics
        batch_mean = x.mean(dim=[0, 1])
        batch_var = x.var(dim=[0, 1])
        
        if self.num_batches_tracked == 0:
            # First batch - initialize directly
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            # Update with momentum
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
    
    def clear_cache(self):
        """Clear the internal cache"""
        self._cache_mean = None
        self._cache_std = None
        self._initialized = torch.tensor(False)
    
    def get_cache_state(self):
        """Get current cache state for saving/loading"""
        return {
            'mean': self._cache_mean,
            'std': self._cache_std,
            'initialized': self._initialized
        }
    
    def set_cache_state(self, state_dict):
        """Set cache state from saved state"""
        self._cache_mean = state_dict['mean']
        self._cache_std = state_dict['std']
        self._initialized = state_dict['initialized']

class ReversibleLogEnergyNorm(nn.Module):
    """
    Reversible logarithmic compression with energy preservation
    Based on: "RealNVP: Real-valued Non-Volume Preserving Flows" (Dinh et al., 2016)
    """
    
    def __init__(self, alpha=0.1, eps=1e-8):
        super().__init__()
        """
        alpha: Compression factor (must be > 0)
        eps: Small constant for numerical stability
        """
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.eps = eps
        
    def forward(self, x, reverse=False, cache=None):
        if not reverse:
            return self._compress_forward(x, cache)
        else:
            return self._compress_reverse(x, cache)
    
    def _compress_forward(self, x, cache=None):
        # Store original sign and magnitude
        sign = torch.sign(x)
        magnitude = torch.abs(x) + self.eps  # Avoid log(0)
        
        # Compute energy for conditioning
        energy = torch.norm(magnitude, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        
        # Reversible log compression
        # y = sign(x) * log(1 + α * |x| / energy) / log(1 + α)
        compression_factor = self.alpha / energy
        magnitude_compressed = torch.log1p(magnitude * compression_factor)
        magnitude_compressed = magnitude_compressed / torch.log1p(torch.tensor(self.alpha))
        
        y = sign * magnitude_compressed
        
        # Store for reverse pass
        if cache is not None:
            cache['energy'] = energy.detach().clone()
            cache['sign'] = sign.detach().clone()
        
        # Log determinant (for flow models)
        log_det = -torch.sum(torch.log(compression_factor.squeeze() + self.eps), dim=-1)
        
        return y, log_det
    
    def _compress_reverse(self, y, cache=None):
        if cache is None:
            raise ValueError("Cache required for reverse transformation")
        
        energy = cache['energy']
        sign = cache['sign']
        
        # Reverse log compression
        # |x| = energy * (exp(y * log(1 + α)) - 1) / α
        magnitude_compressed = torch.abs(y)
        magnitude_original = energy * (torch.exp(magnitude_compressed * torch.log1p(torch.tensor(self.alpha))) - 1) / self.alpha
        
        x_original = sign * magnitude_original
        
        return x_original

class ReversibleMultiScaleDCTNorm(nn.Module):
    """
    Reversible multi-scale decomposition using invertible wavelet-like transforms
    Based on: "Invertible Wavelet Transforms" (Mallat, 2009)
    """
    
    def __init__(self, num_scales=4, num_freq_bins=64):
        super().__init__()
        self.num_scales = num_scales
        self.num_freq_bins = num_freq_bins
        
        # Create reversible frequency bands
        self.band_splits = self._create_reversible_bands()
        
        # Separate affine transforms for each band (must be invertible)
        self.affine_transforms = nn.ModuleList([
            ReversibleAffineTransform(band_size) for band_size in self.band_splits
        ])
    
    def _create_reversible_bands(self):
        """Create bands that allow perfect reconstruction"""
        # Use dyadic splitting for perfect reconstruction
        bands = []
        remaining = self.num_freq_bins
        
        for i in range(self.num_scales - 1):
            # Ensure even splitting for reversibility
            band_size = remaining // 2
            bands.append(band_size)
            remaining -= band_size
        
        bands.append(remaining)
        return bands
    
    def forward(self, x, reverse=False, cache=None):
        if not reverse:
            return self._decompose_forward(x, cache)
        else:
            return self._decompose_reverse(x, cache)
    
    def _decompose_forward(self, x, cache=None):
        batch_size, seq_len, _ = x.shape
        
        # Split into frequency bands
        bands = []
        start_idx = 0
        
        for i, band_size in enumerate(self.band_splits):
            end_idx = start_idx + band_size
            band_data = x[:, :, start_idx:end_idx]
            
            # Apply reversible affine transform to each band
            band_transformed, log_det_band = self.affine_transforms[i](
                band_data, reverse=False
            )
            bands.append(band_transformed)
            
            if i == 0:
                total_log_det = log_det_band
            else:
                total_log_det += log_det_band
            
            start_idx = end_idx
        
        # Concatenate all bands
        y = torch.cat(bands, dim=-1)
        
        # Store band information for reverse pass
        if cache is not None:
            cache['band_splits'] = self.band_splits
        
        return y, total_log_det
    
    def _decompose_reverse(self, y, cache=None):
        if cache is None:
            raise ValueError("Cache required for reverse transformation")
        
        band_splits = cache['band_splits']
        bands_reversed = []
        start_idx = 0
        
        # Reverse each band separately
        for i, band_size in enumerate(band_splits):
            end_idx = start_idx + band_size
            band_data = y[:, :, start_idx:end_idx]
            
            # Reverse affine transform
            band_original = self.affine_transforms[i](
                band_data, reverse=True
            )
            bands_reversed.append(band_original)
            start_idx = end_idx
        
        # Concatenate to reconstruct original
        x_original = torch.cat(bands_reversed, dim=-1)
        return x_original


class ReversibleAffineTransform(nn.Module):
    """Simple reversible affine transformation for each frequency band"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x, reverse=False):
        if not reverse:
            # Forward: y = x * exp(scale) + shift
            y = x * torch.exp(self.scale.view(1, 1, -1)) + self.shift.view(1, 1, -1)
            log_det = torch.sum(self.scale) * x.size(0) * x.size(1)  # For flow models
            return y, log_det
        else:
            # Reverse: x = (y - shift) * exp(-scale)
            x = (x - self.shift.view(1, 1, -1)) * torch.exp(-self.scale.view(1, 1, -1))
            return x
        
class PlaceholderNorm(nn.Module):
    """
    Placeholder normalization layer that does nothing.
    Used when no normalization is desired.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, reverse=False, cache=None):
        return x