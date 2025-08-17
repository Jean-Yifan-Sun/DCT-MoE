import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
import logging

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
# ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MoH_Attention(nn.Module):
    """
    Implements the Mixture-of-Head (MoH) Attention Layer.
    Based on the paper "MoH: Multi-Head Attention as Mixture-of-Head Attention" [cite: 2]
    and the official GitHub implementation.
    """
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None,
                 num_shared_heads=4, top_k=8):
        super().__init__()
        # Standard Attention Parameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim) # Final output projection

        # MoH Specific Parameters
        assert top_k <= (num_heads - num_shared_heads), "top_k must be smaller than the number of routed heads"
        self.num_shared_heads = num_shared_heads
        self.num_routed_heads = num_heads - num_shared_heads
        self.top_k = top_k

        # --- Router Linear Layers ---
        # Corresponds to W_h in the paper (Eq. 6) [cite: 173]
        self.router_head_type = nn.Linear(dim, 2) # 2 for [shared_group, routed_group]

        # Corresponds to W_s in the paper (Eq. 5) [cite: 158]
        self.router_shared = nn.Linear(dim, self.num_shared_heads)
        
        # Corresponds to W_r in the paper (Eq. 5) [cite: 158]
        self.router_routed = nn.Linear(dim, self.num_routed_heads)

    def forward(self, x):
        B, N, C = x.shape # Batch, Sequence Length, Channels

        # 1. Standard QKV projection, shared across all heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- Router Logic ---
        # For ViT, routing decisions are based on the CLS token's query [q[:,:,0,:]]
        # to get a global routing policy for all tokens in the sequence.
        q_for_route = q[:, 0, 0, :] # Shape: [B, C]

        # Stage 1: Route between shared and routed groups
        alpha = F.softmax(self.router_head_type(q_for_route), dim=-1) # Shape: [B, 2]
        alpha_shared = alpha[:, 0].unsqueeze(-1) # Shape: [B, 1]
        alpha_routed = alpha[:, 1].unsqueeze(-1) # Shape: [B, 1]

        # Stage 2: Calculate scores for individual heads within each group
        score_shared = F.softmax(self.router_shared(q_for_route), dim=-1) # Shape: [B, num_shared_heads]
        score_routed = F.softmax(self.router_routed(q_for_route), dim=-1) # Shape: [B, num_routed_heads]

        # Apply Top-K selection for routed heads
        topk_scores, topk_indices = torch.topk(score_routed, self.top_k, dim=-1)

        # Create a sparse mask to zero out non-selected routed heads
        mask_routed = torch.zeros_like(score_routed)
        mask_routed.scatter_(1, topk_indices, 1)
        
        # Combine scores to get final gating weights for all heads
        gating_score_shared = alpha_shared * score_shared
        gating_score_routed = alpha_routed * score_routed * mask_routed
        
        gating_score = torch.cat([gating_score_shared, gating_score_routed], dim=-1) # Shape: [B, num_heads]
        
        # --- Attention & Combination Logic ---
        # Standard scaled dot-product attention is computed for all heads in parallel
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Apply gating scores as a weighted sum over the head dimension
        # Reshape for broadcasting: [B, 1, num_heads, 1] for x_heads [B, N, num_heads, head_dim] is not needed in the GitHub code
        # The GitHub code reshapes x to [B, N, C] then projects. Let's follow that.
        # But the paper implies weighting before projection. Let's implement the paper's formula.
        x_heads = x.reshape(B, N, self.num_heads, self.head_dim)
        gating_reshaped = gating_score.reshape(B, 1, self.num_heads, 1)
        
        # Weighted sum of head outputs 
        x = (x_heads * gating_reshaped).sum(dim=2) # Summing across the heads dimension

        # Final output projection
        x = self.proj(x)

        # --- Load Balance Loss Calculation ---
        # Calculated only for the routed heads [cite: 175]
        # f_i is the fraction of tokens that select a head
        f_i = mask_routed.mean(0)
        # P_i is the average routing probability for a head
        P_i = score_routed.mean(0)
        
        load_balance_loss = torch.sum(f_i * P_i)
        
        return x, load_balance_loss

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Expert(nn.Module):
    """ A single expert in a Mixture of Experts layer. Replace the FFN in the Transformer block with this."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TopKRouter(nn.Module):
    """
    Implements a Top-K Gating function for a Mixture of Experts layer.
    
    This router determines which experts should process each token and calculates
    the weights for combining their outputs. It also computes a load balancing
    loss to encourage even distribution of tokens across experts during training.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, noise_eps: float = 1e-2):
        """
        Args:
            d_model (int): The hidden dimension of the input tokens.
            num_experts (int): The total number of experts available.
            top_k (int): The number of experts to route each token to.
            noise_eps (float): The scaling factor for the noise added during training.
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_eps = noise_eps

        # Learnable linear layer to compute gating logits from token embeddings.
        # This is the W_g matrix mentioned in the survey. 
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            tuple: A tuple containing:
                - final_weights (torch.Tensor): The weights for the experts, of shape [batch_size, seq_len, num_experts].
                - expert_indices (torch.Tensor): The indices of the selected experts, of shape [batch_size, seq_len, top_k].
                - aux_loss (torch.Tensor): The auxiliary load balancing loss.
        """
        # Reshape the input to treat each token independently
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model) # Shape: [batch_size * seq_len, d_model]
        num_tokens = x_flat.shape[0]

        # 1. Calculate Gating Logits
        # Shape: [num_tokens, num_experts]
        logits = self.gate(x_flat)

        # 2. Add Noise (during training) to encourage expert exploration 
        if self.training and self.noise_eps > 0:
            noise = torch.randn_like(logits) * self.noise_eps
            logits += noise

        # 3. Select Top-K Experts
        # Get the scores and indices of the top 'k' experts for each token
        # topk_logits shape: [num_tokens, top_k], topk_indices shape: [num_tokens, top_k]
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # Create a sparse mask to apply softmax only to the top-k experts
        # We create a mask of zeros and scatter 1s at the locations of the top-k experts.
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, topk_indices, 1)

        # Apply the mask: set non-top-k logits to negative infinity for softmax
        masked_logits = logits.where(mask, torch.tensor(float('-inf')))

        # 4. Calculate Gating Weights (Softmax)
        # Softmax is applied to the masked logits to get the final weights.
        final_weights = F.softmax(masked_logits, dim=-1)

        # 5. Calculate Load Balancing Loss 
        # This implementation follows the simplified loss from Switch Transformer 
        # Calculate the fraction of tokens assigned to each expert (f_i)
        f_i = torch.zeros(self.num_experts, device=x.device)
        # Flatten the indices to get a 1D tensor of size [num_tokens * top_k]
        indices_flat = topk_indices.view(-1)
        
        # Create a source tensor of ones that has the SAME size as the flattened indices
        ones_source = torch.ones_like(indices_flat, dtype=torch.float)
        
        # Use the corrected source tensor in the index_add_ operation
        f_i.index_add_(0, indices_flat, ones_source)

        f_i = f_i / num_tokens

        # Calculate the average routing probability for each expert (Q_i)
        Q_i = final_weights.sum(0) / num_tokens
        
        # The auxiliary loss encourages f_i and Q_i to be uniform.
        aux_loss = self.num_experts * torch.sum(f_i * Q_i)

        return final_weights.view(batch_size, seq_len, -1), topk_indices.view(batch_size, seq_len, -1), aux_loss
    
class Block_MoE(nn.Module):
    """ Transformer block with Mixture of Experts (MoE) layer. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, router_layer=TopKRouter, skip=False, use_checkpoint=False, num_experts=2, top_k=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        
        # --- Recommended Change: Use a single LayerNorm before the MoE layer ---
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # --- MoE components ---
        self.router = router_layer(d_model=dim, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            Expert(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
            for _ in range(num_experts))
            
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint and self.training:
            # Use a lambda function to wrap the call to _forward.
            # This correctly handles the two outputs (x, aux_loss).
            x, aux_loss = torch.utils.checkpoint.checkpoint(
                lambda inp, skp: self._forward(inp, skp), 
                x, 
                skip,
                preserve_rng_state=False # Often set to False for efficiency unless you have specific RNG needs
            )
            return x, aux_loss
        else:
            # If not checkpointing, just call _forward directly.
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        # Optional skip connection logic (remains the same)
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        
        # 1. Attention Block (remains the same)
        # This is the first residual connection
        x = x + self.attn(self.norm1(x))
        
        # --- Start of MoE Logic ---
        # Store the output of the attention block for the second residual connection
        residual = x 
        
        # 2. Normalize before routing
        x = self.norm2(x)
        
        # 3. Route tokens to experts
        # The router returns the gating weights and the crucial auxiliary loss
        gating_weights, expert_indices, aux_loss = self.router(x)
        
        # 4. Dispatch tokens and combine expert outputs
        # Create a tensor to store the final output
        final_output = torch.zeros_like(x)
        
        # Flatten tensors for easier indexing
        flat_x = x.view(-1, x.shape[-1])
        flat_weights = gating_weights.view(-1, self.experts.__len__())
        
        # Get the indices of the top-k experts for each token
        topk_indices_flat = expert_indices.view(-1, expert_indices.shape[-1])

        # Loop through each expert and process the tokens routed to it
        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            token_indices = torch.where(topk_indices_flat == i)[0]
            
            if token_indices.numel() > 0:
                # Get the tokens and their corresponding gating weights
                expert_tokens = flat_x[token_indices]
                expert_weights = flat_weights[token_indices, i].unsqueeze(1)
                
                # Process tokens with the expert and apply the gating weight
                expert_output = expert(expert_tokens) * expert_weights
                
                # Add the weighted expert output to the final output tensor
                final_output.view_as(flat_x).index_add_(0, token_indices, expert_output)

        # 5. Second Residual Connection
        x = residual + final_output
        
        # Return both the final output and the auxiliary loss
        return x, aux_loss

class Block_MoH(nn.Module):
    """ Transformer block with Mixture of Heads (MoH) Attention. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False,
                 # MoH specific arguments
                 num_shared_heads=4, top_k=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # --- MoH Attention Layer ---
        self.attn = MoH_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            num_shared_heads=num_shared_heads, top_k=top_k)
            
        # --- Standard MLP Layer ---
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Expert(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
            
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint and self.training:
            # Checkpointing wrapper for the _forward method
            x, aux_loss = torch.utils.checkpoint.checkpoint(
                lambda inp, skp: self._forward(inp, skp),
                x,
                skip,
                preserve_rng_state=False
            )
            return x, aux_loss
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        
        # --- Start of MoH Logic ---
        # The MoH_Attention layer returns both the output and the aux_loss
        attn_output, aux_loss = self.attn(self.norm1(x))
        
        # 1. First Residual Connection (after attention)
        x = x + attn_output
        
        # 2. Second Residual Connection (after MLP)
        x = x + self.mlp(self.norm2(x))
        
        # Return the final output and the auxiliary loss from the attention layer
        return x, aux_loss

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs

        self.proj = nn.Linear(self.DCT_coes * 6, embed_dim, bias=True)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 6, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.orthogonal_(m.weight)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        x = self.proj(x)  # (b, tokens, num_low_freq*6) --> (b, tokens, hidden_dim)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)  # (b, dim) --> (b, 1, dim)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)  # (b, tokens, dim)
            skips.append(x)

        x = self.mid_block(x)  # (b, tokens, dim)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())  # (b, tokens, dim)

        x = self.norm(x)
        x = self.decoder_pred(x)  # (b, tokens, dim) --> (b, tokens, num_low_freq*6)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]  # (b, tokens, num_low_freq)

        return x

class UViT_greyscale(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs

        # 只用Y通道，输入输出都是 low_freqs*4
        self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1
        

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        # x: (b, tokens, num_low_freq*4)
        x = self.proj(x)  # (b, tokens, num_low_freq*4) --> (b, tokens, hidden_dim)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)  # (b, 1, dim)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)

        image_tokens = x[:, self.extras:, :] # Select the image tokens first
        x = self.decoder_pred(image_tokens) # Then apply the prediction head ONLY to them

        # x = self.decoder_pred(x)  # (b, tokens, dim) --> (b, tokens, num_low_freq*4)
        # assert x.size(1) == self.extras + L
        # x = x[:, self.extras:, :]  # (b, tokens, num_low_freq*4)

        return x

class UViT_greyscale_cond(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs

        # 只用Y通道，输入输出都是 low_freqs*4
        self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True)
        self.label_emb = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True) # nn.Embedding or nn.Linear
        self.extras = 1 + self.tokens  # 1 for time token, self.tokens for label embedding

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x:torch.Tensor, timesteps, y:torch.Tensor):
        # x: (b, tokens, num_low_freq*4)
        # y: (b, tokens, num_low_freq*4)
        
        B, L, D = x.shape

        assert x.size() == y.size(), f'Expected x and y to have the same shape, got {x.size()} and {y.size()}'
        x = self.proj(x)  # (b, tokens, num_low_freq*4) --> (b, tokens, hidden_dim)
        label_emb = self.label_emb(y) # (b, tokens, num_low_freq*4) --> (b, tokens, hidden_dim)
        x = torch.cat((label_emb, x), dim=1)# (b, tokens*2, hidden_dim)

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)  # (b, 1, dim)
        x = torch.cat((time_token, x), dim=1)
        
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)  # (b, tokens, dim) --> (b, tokens, num_low_freq*4)
        assert x.size(1) == self.extras + L, f'Expected x to have shape (b, {self.extras + L}, num_low_freq*4), got {x.size()}'
        x = x[:, self.extras:, :]  # (b, tokens, num_low_freq*4)

        return x

class UViT_greyscale_MoE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=True, MoE={"depth": 1, "num_experts": 2, "router":"topk", "top_k": 2}):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs
        assert use_moe==True, "UViT_greyscale_MoE is designed to use MoE. Set use_moe=True."
        
        # --- MoE Configuration ---
        self.num_experts = MoE.get("num_experts", 2)
        self.router_type = MoE.get("router", "topk")
        self.top_k = MoE.get("top_k", 2) # How many experts to use per token
        self.moe_layer_index = MoE.get("depth", 1) # Interpreted as placing MoE at first and last blocks

        # --- Input and Embedding Layers ---
        self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), 
            nn.SiLU(), 
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1
            
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))
        
        # --- Build Transformer Blocks with MoE ---
        in_blocks_list = []
        for i in range(depth // 2):
            if self.moe_layer_index == 1 and i == 0: # First block
                block = Block_MoE(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, top_k=self.top_k)
            else:
                 block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            in_blocks_list.append(block)
        self.in_blocks = nn.ModuleList(in_blocks_list)

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        out_blocks_list = []
        for i in range(depth // 2):
            if self.moe_layer_index == 1 and i == (depth // 2) - 1: # Last block
                 block = Block_MoE(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, num_experts=self.num_experts, top_k=self.top_k)
            else:
                block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            out_blocks_list.append(block)
        self.out_blocks = nn.ModuleList(out_blocks_list)

        # --- Output Layers ---
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        # 1. Initial Projection and Embedding
        x = self.proj(x)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y).unsqueeze(1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        # 2. Forward pass through the network, tracking aux loss
        total_aux_loss = 0.0
        skips = []

        for blk in self.in_blocks:
            if isinstance(blk, Block_MoE):
                x, aux_loss = blk(x)
                total_aux_loss += aux_loss
            else:
                x = blk(x)
            skips.append(x)

        # Mid block does not have MoE in this design
        x = self.mid_block(x)

        for blk in self.out_blocks:
            if isinstance(blk, Block_MoE):
                x, aux_loss = blk(x, skips.pop())
                total_aux_loss += aux_loss
            else:
                x = blk(x, skips.pop())

        # 3. Final Prediction Head
        x = self.norm(x)
        image_tokens = x[:, self.extras:, :]
        x = self.decoder_pred(image_tokens)

        # Return both the prediction and the accumulated auxiliary loss
        return x, total_aux_loss

class UViT_greyscale_MoH(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=True, MoH={"depth": 1, "num_shared_heads": 4, "top_k": 8}):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs
        assert use_moe==True, "UViT_greyscale_MoH is designed to use MoH. Set use_moe=True."

        # --- MoH Configuration ---
        self.num_shared_heads = MoH.get("num_shared_heads", 4)
        self.top_k = MoH.get("top_k", 8) # How many experts heads to use per token
        self.moh_layer_index = MoH.get("depth", 1) # Interpreted as placing MoH at first and last blocks

        # --- Input and Embedding Layers ---
        self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), 
            nn.SiLU(), 
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1
            
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))
        
        # --- Build Transformer Blocks with MoE ---
        in_blocks_list = []
        for i in range(depth // 2):
            if self.moh_layer_index == 1 and i == 0: # First block
                block = Block_MoH(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_shared_heads=self.num_shared_heads, top_k=self.top_k)
            else:
                 block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            in_blocks_list.append(block)
        self.in_blocks = nn.ModuleList(in_blocks_list)

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        out_blocks_list = []
        for i in range(depth // 2):
            if self.moh_layer_index == 1 and i == (depth // 2) - 1: # Last block
                 block = Block_MoH(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, num_shared_heads=self.num_shared_heads, top_k=self.top_k)
            else:
                block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            out_blocks_list.append(block)
        self.out_blocks = nn.ModuleList(out_blocks_list)

        # --- Output Layers ---
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        # 1. Initial Projection and Embedding
        x = self.proj(x)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y).unsqueeze(1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        # 2. Forward pass through the network, tracking aux loss
        total_aux_loss = 0.0
        skips = []

        for blk in self.in_blocks:
            if isinstance(blk, Block_MoH):
                x, aux_loss = blk(x)
                total_aux_loss += aux_loss
            else:
                x = blk(x)
            skips.append(x)

        # Mid block does not have MoE in this design
        x = self.mid_block(x)

        for blk in self.out_blocks:
            if isinstance(blk, Block_MoH):
                x, aux_loss = blk(x, skips.pop())
                total_aux_loss += aux_loss
            else:
                x = blk(x, skips.pop())

        # 3. Final Prediction Head
        x = self.norm(x)
        image_tokens = x[:, self.extras:, :]
        x = self.decoder_pred(image_tokens)

        # Return both the prediction and the accumulated auxiliary loss
        return x, total_aux_loss