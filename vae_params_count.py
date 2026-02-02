import torch
from diffusers import AutoencoderKL

# 1. Load the VAE
# Note: You need a Hugging Face token with access to FLUX.1-dev
vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="vae"
)

# 2. Calculate Total Parameters
total_params = sum(p.numel() for p in vae.parameters())

# 3. Calculate Trainable Parameters (in case some are frozen)
trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

print(f"Total VAE Parameters: {total_params:,}")
print(f"Trainable VAE Parameters: {trainable_params:,}")

# Optional: Estimate size in VRAM (FP16)
# 2 bytes per parameter for FP16
print(f"Approximate File Size (FP16): {total_params * 2 / 1024**2:.2f} MB")