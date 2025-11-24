import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
from absl import logging
import numpy as np
# from normalization import *

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
    Based on the paper "MoH: Multi-Head Attention as Mixture-of-Head Attention"
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, 
                 attn_drop=0., proj_drop=0., shared_head=0, routed_head=0,
                 load_balance_lambda=1, usage_decay=0.9):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert routed_head <= num_heads - shared_head, "routed_head must be <= num_heads - shared_head"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.load_balance_lambda = load_balance_lambda
        self.usage_decay = usage_decay
        
        # Temperature parameter for attention scaling with better initialization
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1) * 0.5)  # More stable initialization

        # Query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # MoH routing parameters
        self.shared_head = shared_head
        self.routed_head = routed_head
        
        if self.routed_head > 0:
            self.wg = nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = nn.Linear(dim, shared_head, bias=False)

        # For tracking head usage - fixed size to match actual number of routable heads
        self.register_buffer('head_usage', torch.zeros(num_heads - shared_head))
        self.register_buffer('num_updates', torch.zeros(1))

    def forward(self, x):
        B, N, C = x.shape
        _x = x.reshape(B * N, C)
        
        # Initialize load balancing loss
        l_aux = torch.tensor(0.0, device=x.device)

        # MoH routing mechanism
        if self.routed_head > 0:
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)

            # Straight-Through Estimator for topk
            num_tokens, num_experts = gates.shape
            
            # Get top-k indices and create hard mask
            topk_values, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask_hard = F.one_hot(indices, num_classes=num_experts).sum(dim=1).float()
            
            # Create soft mask for backward pass using STE
            mask_soft = mask_hard.detach() - gates.detach() + gates
            
            if self.training:
                # Calculate load balancing loss
                me = gates.mean(dim=0)
                ce = mask_hard.mean(dim=0)  # Use hard mask for accurate statistics
                l_aux = self.load_balance_lambda * torch.sum(me * ce) * num_experts
                
                # Update head usage statistics
                current_usage = mask_hard.mean(dim=0)
                self.num_updates += 1
                self.head_usage = (self.usage_decay * self.head_usage + 
                                  (1 - self.usage_decay) * current_usage.detach())

            # Use soft mask for routing (STE ensures gradients flow to gates)
            routed_head_gates = gates * mask_soft
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        # Compute queries, keys, values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate attention with scaled cosine attention and query embedding
        # Add stability epsilon to normalization
        q_normalized = F.normalize(q, dim=-1, eps=1e-6) + self.query_embedding
        k_normalized = F.normalize(k, dim=-1, eps=1e-6)
        
        # Clamp temperature to prevent extreme values
        temperature = torch.clamp(F.softplus(self.temperature), min=1e-3, max=1e3)
        attn = (q_normalized * temperature) @ k_normalized.transpose(-2, -1)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values and combine heads with routing gates
        if self.routed_head > 0:
            x = (attn @ v).transpose(1, 2)  # B, N, head, dim

            if self.shared_head > 1:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            else:
                shared_head_gates = torch.ones((B, N, self.shared_head), device=x.device, dtype=x.dtype) * self.shared_head
            
            if self.shared_head == 0:
                masked_gates = routed_head_gates
            else:
                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
                
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,1], routed_head_gates)

                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, C)
        else:
            # Only shared heads case
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
                    
            x = (attn @ v).transpose(1, 2)  # B, N, head, dim
            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, l_aux

    def get_load_balance_loss(self):
        """Get the load balancing loss for all MoH layers"""
        # This can be used to aggregate losses from multiple MoH layers
        if self.num_updates > 0:
            head_prob = self.head_usage / (self.head_usage.sum() + 1e-8)
            balance_loss = -self.load_balance_lambda * (head_prob * torch.log(head_prob + 1e-8)).sum()
            return balance_loss
        return torch.tensor(0.0, device=self.head_usage.device)

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
        A more robust implementation of the Top-K Gating function with STE.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            tuple: A tuple containing:
                - final_weights (torch.Tensor): The weights for the experts, of shape [batch_size, seq_len, num_experts].
                - expert_indices (torch.Tensor): The indices of the selected experts, of shape [batch_size, seq_len, top_k].
                - aux_loss (torch.Tensor): The auxiliary load balancing loss.
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model) # Shape: [batch_size * seq_len, d_model]
        num_tokens = x_flat.shape[0]

        # 1. Calculate Gating Logits and Probabilities
        logits = self.gate(x_flat)
        if self.training and self.noise_eps > 0:
            noise = torch.randn_like(logits) * self.noise_eps
            logits += noise
        
        # 1. 移除极端值
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        # 2. 减去最大值进行数值稳定化
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0].detach()
        logits = logits - logits_max
        # router_probs are the soft probabilities used for the backward pass (gradient calculation)
        router_probs = F.softmax(logits.float(), dim=-1)

        # 2. Select Top-K Experts and create a hard mask for the forward pass
        # This avoids division by small numbers and is more stable.
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        hard_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        hard_mask = hard_mask.float()

        # 3. Apply Straight-Through Estimator (STE)
        # For the backward pass, we want gradients to flow through the original router_probs,
        # but for the forward pass, we use the hard 0/1 mask.
        ste_weights = (hard_mask - router_probs).detach() + router_probs

        # 4. Calculate Load Balancing Loss
        # This loss encourages tokens to be distributed evenly across experts.
        # f_i: Fraction of tokens dispatched to expert i (using the hard mask for accuracy).
        # P_i: Average router probability for expert i.
        f_i = ste_weights.sum(0) / num_tokens
        P_i = router_probs.sum(0) / num_tokens
        # 避免数值不稳定
        f_i = torch.clamp(f_i, min=1e-6)
        P_i = torch.clamp(P_i, min=1e-6)
        
        # The loss is the dot product of these two vectors, scaled by the number of experts.
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        aux_loss = torch.clamp(aux_loss, max=100.0)  # 防止损失爆炸

        return ste_weights.view(batch_size, seq_len, -1), topk_indices.view(batch_size, seq_len, -1), aux_loss
    
    def forward_old(self, x: torch.Tensor):
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

        # 3. 计算路由概率（可微部分）
        router_probs = F.softmax(logits, dim=-1)
        
        # 4. 使用STE处理topk操作
        # Get the scores and indices of the top 'k' experts for each token
        # topk_logits shape: [num_tokens, top_k], topk_indices shape: [num_tokens, top_k]
        # 前向传播：使用硬选择（不可微）
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 创建硬掩码
        hard_mask = torch.zeros_like(router_probs)
        hard_mask.scatter_(1, topk_indices, 1.0)
        
        # 反向传播：使用软概率（可微）
        # 这是STE的关键部分 - 在前向中使用硬掩码，但在反向中使用软概率
        soft_mask = hard_mask - router_probs.detach() + router_probs
        
        # 5. 计算最终权重
        # 使用硬掩码进行前向计算，但梯度通过软掩码回传
        final_weights = soft_mask
        
        # 6. 计算辅助损失（使用STE处理不可微部分）
        # 计算硬分配的f_i（用于前向）
        f_i_hard = torch.zeros(self.num_experts, device=x.device)
        indices_flat = topk_indices.view(-1)
        ones_source = torch.ones_like(indices_flat, dtype=torch.float)
        f_i_hard.index_add_(0, indices_flat, ones_source)
        f_i_hard = f_i_hard / (num_tokens * self.top_k)
        
        # 计算软分配的f_i（用于反向）
        # 使用路由概率的均值作为软分配的f_i
        f_i_soft = router_probs.mean(0)
        
        # 使用STE处理f_i计算
        f_i = f_i_hard - f_i_soft.detach() + f_i_soft
        
        # 计算Q_i
        Q_i = final_weights.sum(0) / num_tokens
        
        # 计算辅助损失
        aux_loss = self.num_experts * torch.sum(f_i * Q_i)
        
        return final_weights.view(batch_size, seq_len, -1), topk_indices.view(batch_size, seq_len, -1), aux_loss

class ECDiTRoutingLayer(nn.Module):
    """
    EC-DiT Routing Layer: Expert-Choice Routing for Diffusion Transformers
    """

    def __init__(self, dim, num_experts, expert_capacity_factor=2.0, mlp_hidden_dim=2048, act_layer=nn.GELU, num_tokens=1, counting=False):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = expert_capacity_factor
        
        # Router parameters (expert embeddings)
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert networks (FeedForward networks)
        self.experts = nn.ModuleList([
            Expert(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer) for _ in range(num_experts)
        ])
        # 添加 buffer 来记录 token 分配情况
        # Shape: [num_tokens] - 每个 token 被多少专家选中
        self.counting = True
        self.register_buffer('token_selection_count', torch.zeros(num_experts, num_tokens, dtype=torch.float))
        self.decay = 0.99  # 指数移动平均的衰减率
        
        
    def forward(self, x_prime):
        """
        x_prime: [batch_size, seq_len, dim] - input after cross-attention
        Returns: [batch_size, seq_len, dim] - routed output
        """
        batch_size, seq_len, dim = x_prime.shape
        
        # Step 1: Compute token-expert affinity scores
        # A = softmax(x' @ W_r) - equation (5) in paper
        affinity_scores = self.router(x_prime)  # [B, S, E]
        affinity_scores = F.softmax(affinity_scores, dim=-1)  # [B, S, E]
        
        # Step 2: Calculate expert capacity
        # C = S × f_c / E - capacity per expert
        expert_capacity = int(seq_len * self.capacity_factor / self.num_experts) + 1
        expert_capacity = max(1, expert_capacity)  # Ensure at least 1
        
        # Step 3: Create gating tensor G using expert-choice routing
        gating_tensor, selection_indices, selection_mask = self._expert_choice_routing(affinity_scores, expert_capacity)

        # 更新 token 分配统计
        if self.training and self.counting:
            with torch.no_grad():
                # 计算每个 token 被选中的次数
                token_counts = torch.zeros(self.num_experts, seq_len, device=x_prime.device)
                token_score = torch.ones(self.num_experts, seq_len, device=x_prime.device, dtype=torch.float)
                for _ in range(batch_size):
                    for i in range(self.num_experts):
                    # selection_indices: [B, E, C] -> [B*E*C]
                        expert_selections = selection_indices[_, i, :]  # [C]
                        
                    # 累加每个 token 被选中的次数
                        token_counts[i, expert_selections] += token_score[i, expert_selections]
                
                # 归一化: 除以专家总数
                token_counts = token_counts / (batch_size)
                
                # 使用指数移动平均更新统计
                # if self.token_selection_count.shape != token_counts.shape:
                #     self.token_selection_count = token_counts
                # else:
                self.token_selection_count = (
                        self.token_selection_count * self.decay + 
                        token_counts * (1 - self.decay)
                    )
        
        # Step 4: Process tokens through experts and combine outputs
        output = self._apply_experts(x_prime, gating_tensor, selection_indices, selection_mask, expert_capacity)
        
        return output
    
    def _expert_choice_routing(self, affinity_scores, expert_capacity):
        """
        Expert-choice routing: Each expert selects top-C tokens
        Corresponds to equation (6) in paper
        """
        batch_size, seq_len, num_experts = affinity_scores.shape
        
        # Reshape to [B, E, S] for expert-wise operations
        affinity_scores_t = affinity_scores.transpose(1, 2)  # [B, E, S]
        
        # Find top-C tokens for each expert
        topk_values, topk_indices = torch.topk(
            affinity_scores_t, 
            k=expert_capacity, 
            dim=-1
        )  # [B, E, C]
        
        # Create gating tensor initialized to zeros
        # Create selection mask and indices for gathering
        selection_indices = topk_indices  # [B, E, C]
        selection_mask = torch.ones_like(topk_values, dtype=torch.bool)  # [B, E, C]
        gating_tensor = torch.zeros_like(affinity_scores_t)  # [B, E, S]
        
        # Scatter top-k values to appropriate positions
        batch_indices = torch.arange(batch_size, device=affinity_scores.device)
        batch_indices = batch_indices.view(-1, 1, 1).expand(-1, num_experts, expert_capacity)
        
        expert_indices = torch.arange(num_experts, device=affinity_scores.device)
        expert_indices = expert_indices.view(1, -1, 1).expand(batch_size, -1, expert_capacity)
        
        # Use scatter_ to assign values
        gating_tensor.scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_values
        )
        
        # Transpose back to [B, S, E]
        gating_tensor = gating_tensor.transpose(1, 2)  # [B, S, E]
        
        return gating_tensor, selection_indices, selection_mask
    
    def _apply_experts(self, x_prime, gating_tensor, selection_indices, selection_mask, expert_capacity):
        """
        Apply experts to selected tokens and combine results
        Corresponds to equation (8) in paper
        """
        batch_size, seq_len, dim = x_prime.shape
        num_experts = len(self.experts)

        # Initialize output
        output = torch.zeros_like(x_prime)  # [B, S, D]
        
        # Process each expert separately (clearer and more debuggable)
        for expert_idx in range(num_experts):
            expert_fn = self.experts[expert_idx]
            
            # Get the tokens selected by this expert
            # selection_indices: [B, E, C] -> for this expert: [B, C]
            expert_token_indices = selection_indices[:, expert_idx, :]  # [B, C]
            
            # Get gating values for these tokens
            expert_gating_values = torch.gather(
                gating_tensor[:, :, expert_idx],  # [B, S]
                dim=1,
                index=expert_token_indices  # [B, C]
            ).unsqueeze(-1)  # [B, C, 1]
            
            # Gather the actual tokens to process
            # Create batch indices
            batch_indices = torch.arange(batch_size, device=x_prime.device)
            batch_indices = batch_indices.view(-1, 1).expand(batch_size, expert_capacity)  # [B, C]
            
            # Gather tokens: [B, C, D]
            selected_tokens = x_prime[batch_indices, expert_token_indices, :]
            
            # Process through expert
            # Flatten batch and capacity for processing
            selected_tokens_flat = selected_tokens.reshape(-1, dim)  # [B*C, D]
            expert_output_flat = expert_fn(selected_tokens_flat)  # [B*C, D]
            expert_output = expert_output_flat.reshape(batch_size, expert_capacity, dim)  # [B, C, D]
            
            # Apply gating
            gated_output = expert_output * expert_gating_values  # [B, C, D]
            
            # Scatter back to output - FIXED DIMENSION HANDLING
            # We need to scatter along dimension 1 (sequence dimension)
            # output: [B, S, D]
            # expert_token_indices: [B, C] 
            # gated_output: [B, C, D]
            
            # Use scatter_add_ with proper dimensions
            output.scatter_add_(
                dim=1,  # scatter along sequence dimension
                index=expert_token_indices.unsqueeze(-1).expand(-1, -1, dim),  # [B, C, D]
                src=gated_output  # [B, C, D]
            )
        
        return output
    
    def get_expert_distribution(self):
        """获取专家选择的分布统计"""
        return self.token_selection_count.cpu().numpy()

class Block_MoE(nn.Module):
    """ Transformer block with Mixture of Experts (MoE) layer. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, router_layer=TopKRouter, skip=False, use_checkpoint=False, num_experts=2, top_k=2, noise_eps=1e-2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        
        # --- Recommended Change: Use a single LayerNorm before the MoE layer ---
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # --- MoE components ---
        self.router = router_layer(d_model=dim, num_experts=num_experts, top_k=top_k, noise_eps=noise_eps)
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
        
        # 1. Attention Block
        x = x + self.attn(self.norm1(x))
        residual = x
        
        # 2. MoE Block
        x = self.norm2(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.shape[-1])
        
        # 3. 获取路由权重和专家分配
        try:
            gating_weights, expert_indices, aux_loss = self.router(x)
            
            # 检查路由权重的有效性
            # if not torch.all(torch.isfinite(gating_weights)):
            #     logging.error(f"非有效的路由权重: {gating_weights.min()}, {gating_weights.max()}")
            #     gating_weights = torch.nan_to_num(gating_weights, nan=0.0, posinf=1.0, neginf=0.0)
            #     gating_weights = F.normalize(gating_weights, p=1, dim=-1)
            
            flat_weights = gating_weights.view(-1, self.experts.__len__())
            topk_indices_flat = expert_indices.view(-1, expert_indices.shape[-1])
            
            # 4. 专家处理
            expert_outputs = []
            expert_masks = []
            
            for i, expert in enumerate(self.experts):
                # 找出被路由到当前专家的token
                token_indices = torch.where(topk_indices_flat == i)[0]
                
                if token_indices.numel() > 0:
                    # 获取相关的token和权重
                    expert_tokens = flat_x[token_indices]
                    expert_weights = flat_weights[token_indices, i].unsqueeze(1)
                    
                    # 对专家输出进行安全处理
                    with torch.set_grad_enabled(True):  # 确保梯度流动
                        expert_output = expert(expert_tokens)
                        # 检查专家输出
                        expert_output = torch.clamp(expert_output, min=-1e6, max=1e6)

                        # 创建sparse更新掩码
                        expert_mask = torch.zeros_like(flat_x)
                        expert_mask[token_indices] = expert_weights
                        expert_masks.append(expert_mask)
                        
                        # 应用权重并累积输出
                        weighted_output = expert_output * expert_weights
                        expert_outputs.append((token_indices, weighted_output))
            
            # 5. 安全地组合专家输出
            for token_indices, weighted_output in expert_outputs:
                final_output.view_as(flat_x).index_add_(0, token_indices, weighted_output)
            
            # 确保输出不包含极端值
            # if not torch.all(torch.isfinite(final_output)):
            #     logging.error("MoE输出包含非有效值")
            #     final_output = torch.nan_to_num(final_output, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 6. 残差连接
            x = residual + final_output
            
            # 返回结果和辅助损失
            return x, aux_loss
            
        except Exception as e:
            logging.error(f"MoE前向传播错误: {str(e)}")
            # logging.error(f"堆栈跟踪: {traceback.format_exc()}")
            raise e

class Block_LH_MoE(nn.Module):
    """ Transformer block with Mixture of Experts (MoE) layer. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, router_layer=TopKRouter, skip=False, use_checkpoint=False, num_experts=2, per_expert_emblength=1):
        super().__init__()
        self.num_experts = num_experts
        self.per_expert_emblength = per_expert_emblength
        block_list = []
        for i in range(self.num_experts):
            block = Block(
                    dim=per_expert_emblength, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            block_list.append(block)
        self.blocks = nn.ModuleList(block_list)
            
    def forward(self, x, skip=None):
        x_list = []
        if skip is None:
            for i in range(self.num_experts):
                x_i =  x[:, :, i * self.per_expert_emblength : (i + 1) * self.per_expert_emblength]
                x_i = self.blocks[i](x_i)
                if i == 0:
                    x_ = x_i
                else:
                    x_ = torch.cat((x_, x_i), dim=-1) # Concatenate along the embedding dimension
        else:
            for i in range(self.num_experts):
                skip_i = skip[:, :, i * self.per_expert_emblength : (i + 1) * self.per_expert_emblength]
                x_i =  x[:, :, i * self.per_expert_emblength : (i + 1) * self.per_expert_emblength]
                x_i = self.blocks[i](x_i, skip_i)
                if i == 0:
                    x_ = x_i
                else:
                    x_ = torch.cat((x_, x_i), dim=-1) # Concatenate along the embedding dimension
        return x_


class Block_MoH(nn.Module):
    """ Transformer block with Mixture of Heads (MoH) Attention. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, attn_drop=0, proj_drop=0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False,
                 # MoH specific arguments
                 num_shared_heads=4, top_k=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # --- MoH Attention Layer ---
        self.attn = MoH_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            shared_head=num_shared_heads, routed_head=top_k)
            
        # --- Standard MLP Layer ---
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
            
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

class Block_ECDiT(nn.Module):
    """ Transformer block with EC-DiT Routing Layer. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, num_experts=2, expert_capacity_factor=1.0, num_tokens=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        
        # --- Recommended Change: Use a single LayerNorm before the MoE layer ---
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # --- EC-DiT components ---
        self.routing_layer = ECDiTRoutingLayer(dim=dim, num_experts=num_experts, expert_capacity_factor=expert_capacity_factor, mlp_hidden_dim=mlp_hidden_dim, act_layer=act_layer, num_tokens=num_tokens)
            
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(lambda inp, skp: self._forward(inp, skp), x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        
        # 1. Attention Block
        x = x + self.attn(self.norm1(x))
        residual = x
        
        # 2. EC-DiT Routing Layer
        x = self.norm2(x)
        x = self.routing_layer(x)
        
        # 3. Residual Connection
        x = residual + x
        
        return x
    
    def get_expert_distribution(self):
        """获取专家选择的分布统计"""
        return self.routing_layer.get_expert_distribution()

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=False):
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
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=False, MoE=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs

        if in_chans == 1:
            self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True)
            self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)
        else:
            self.proj = nn.Linear(in_chans, embed_dim, bias=True)
            self.decoder_pred = nn.Linear(embed_dim, in_chans, bias=True)

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
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=True, MoE={"typr":'normal', "depth": 1, "num_experts": 2, "router":"topk", "top_k": 2},
                 pos_normalize="minmax"):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs
        assert use_moe==True, "UViT_greyscale_MoE is designed to use MoE. Set use_moe=True."
        # self.pos_normalize = False
        
        # --- MoE Configuration ---
        self.moe_type = MoE.get("type", "normal") # 'normal', 'ecmoe'
        self.num_experts = MoE.get("num_experts", 2)
        self.router_type = MoE.get("router", "topk")
        self.top_k = MoE.get("top_k", 2) # How many experts to use per token
        self.moe_noise_eps = MoE.get("noise_eps", 1e-2)
        self.moe_layer_index = MoE.get("depth", 1) # Interpreted as placing MoE at first and last blocks

        # --- Input and Embedding Layers ---
        if in_chans != 1:
            self.proj = nn.Linear(in_chans, embed_dim, bias=True)
        else:
            self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True) # For greyscale images, only use Y channel
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
            if self.moe_layer_index == i + 1 or self.moe_layer_index == -1: # First block or all blocks
                if self.moe_type == 'ecmoe':
                    block = Block_ECDiT(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, expert_capacity_factor=self.top_k, num_tokens=self.tokens + self.extras)
                elif self.moe_type == 'normal':
                    block = Block_MoE(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, top_k=self.top_k, noise_eps=self.moe_noise_eps)
            else:
                 block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            in_blocks_list.append(block)
        self.in_blocks = nn.ModuleList(in_blocks_list)
        
        if self.moe_layer_index == -1:
            if self.moe_type == 'ecmoe':
                self.mid_block = Block_ECDiT(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, expert_capacity_factor=self.top_k, num_tokens=self.tokens + self.extras)
            elif self.moe_type == 'normal':
                self.mid_block = Block_MoE(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, top_k=self.top_k, noise_eps=self.moe_noise_eps)
        else:    
            self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        out_blocks_list = []
        for i in range(depth // 2):
            if self.moe_layer_index + i == (depth // 2) or self.moe_layer_index == -1: # Last block
                if self.moe_type == 'ecmoe':
                    block = Block_ECDiT(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, num_experts=self.num_experts, expert_capacity_factor=self.top_k, num_tokens=self.tokens + self.extras) 
                elif self.moe_type == 'normal':
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
        if in_chans != 1:
            self.decoder_pred = nn.Linear(embed_dim, in_chans, bias=True)
        else:
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
        # if self.pos_normalize:
        #     x = self.input_normalize(x)

        x = self.proj(x)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y).unsqueeze(1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        # 2. Forward pass through the network, tracking aux loss
        total_aux_loss = torch.tensor(0.0).to(x.device)
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

        # if self.pos_normalize:
        #     x = self.input_normalize(x, reverse=True)

        # Return both the prediction and the accumulated auxiliary loss
        return x, total_aux_loss
    
    def get_expert_distribution(self):
        """获取所有MoE层的专家选择的分布统计"""
        distributions = {}
        for i, blk in enumerate(self.in_blocks):
            if isinstance(blk, Block_ECDiT):
                distributions[f'in_block_{i}'] = blk.get_expert_distribution()
        for i, blk in enumerate(self.out_blocks):
            if isinstance(blk, Block_ECDiT):
                distributions[f'out_block_{i}'] = blk.get_expert_distribution()
        return distributions

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
        if in_chans != 1:
            self.proj = nn.Linear(in_chans, embed_dim, bias=True)
        else:
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
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
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
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, num_shared_heads=self.num_shared_heads, top_k=self.top_k)
            else:
                block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            out_blocks_list.append(block)
        self.out_blocks = nn.ModuleList(out_blocks_list)

        # --- Output Layers ---
        self.norm = norm_layer(embed_dim)
        if in_chans != 1:
            self.decoder_pred = nn.Linear(embed_dim, in_chans,bias=True)
        else:
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


class UViT_greyscale_LH_MoE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=True, MoE={"depth": 1, "num_experts": 2, "router":"topk", "top_k": 2}):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs
        assert use_moe==True, "UViT_greyscale_MoE is designed to use MoE. Set use_moe=True."
        
        # --- LH_MoE Configuration ---
        self.num_experts = MoE.get("num_experts", 2)
        assert self.DCT_coes * 4 % self.num_experts == 0, f"num_experts ({self.num_experts}) must divide DCT_coes*4 ({self.DCT_coes * 4}) evenly for LH_MoE."
        assert self.embed_dim % self.num_experts == 0, f"num_experts ({self.num_experts}) must divide embed_dim ({self.embed_dim}) evenly for LH_MoE."
        self.per_expert_dctlength = self.DCT_coes * 4 // self.num_experts
        self.per_expert_emblength = self.embed_dim // self.num_experts
        self.router_type = MoE.get("router", "frequency_avg")
        # self.top_k = MoE.get("top_k", 2) # How many experts to use per token
        # self.moe_noise_eps = MoE.get("noise_eps", 1e-2)
        self.moe_layer_index = MoE.get("depth", 1) # Interpreted as placing MoE at first and last blocks

        # --- Input and Embedding Layers ---
        self.proj = nn.ModuleList()
        for i in range(self.num_experts):
            self.proj.append(nn.Linear(self.per_expert_dctlength, self.per_expert_emblength, bias=True))
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
                block = Block_LH_MoE(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, per_expert_emblength=self.per_expert_emblength)
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
                 block = Block_LH_MoE(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, num_experts=self.num_experts, per_expert_emblength=self.per_expert_emblength)
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
        for i in range(self.num_experts):
            x_part = x[:, :, i * self.per_expert_dctlength : (i + 1) * self.per_expert_dctlength]
            x_emb_part = self.proj[i](x_part) # (b, tokens, per_expert_emblength)
            if i == 0:
                x_ = x_emb_part
            else:
                x_ = torch.cat((x_, x_emb_part), dim=-1) # Concatenate along the embedding dimension

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x_), dim=1)
        if y is not None:
            label_emb = self.label_emb(y).unsqueeze(1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        # 2. Forward pass through the network, tracking aux loss
        total_aux_loss = torch.tensor(0.0, device=x.device)
        skips = []

        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        # Mid block does not have MoE in this design
        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        # 3. Final Prediction Head
        x = self.norm(x)
        image_tokens = x[:, self.extras:, :]
        x = self.decoder_pred(image_tokens)

        # Return both the prediction and the accumulated auxiliary loss
        return x, total_aux_loss

