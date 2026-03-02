import math
import copy
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# --------------------------------------------------------------------------- #
# 1. Utilities
# --------------------------------------------------------------------------- #
def compute_out_len(L: int, k: int, s: int) -> int:
    """Compute output length with front padding (causal padding)"""
    return ceil_div(L, s)

def pick(val, idx: int) -> int:
    '''Pick Kernel size for each stage'''
    return int(val[idx]) if isinstance(val, (list, tuple)) else int(val)

def ceil_div(a: int, b: int) -> int:
    """Ceiling division: returns ⌈a/b⌉"""
    return (a + b - 1) // b

# --------------------------------------------------------------------------- #
# 2. Stem padding
# --------------------------------------------------------------------------- #
class FrontPadding(nn.Module):
    def __init__(self, patch_size: int, stride: int):
        super().__init__()
        self.k, self.s = patch_size, stride

    def forward(self, x):
        T = x.size(-1)
        # Calculate padding needed for ceil division
        out_len = ceil_div(T, self.s)
        total_len_needed = (out_len - 1) * self.s + self.k
        pad = max(0, total_len_needed - T)
        if pad == 0:
            return x
        return torch.cat([x[..., -1:].repeat(*(1,)*(x.dim()-1), pad), x], dim=-1)

# --------------------------------------------------------------------------- #
# 3. Core Blocks: ConvMix, FFN
# --------------------------------------------------------------------------- #
class DepthwiseMix(nn.Module):
    def __init__(self, Cin: int, Cout: int, kL: int, kS: int, bias: bool):
        super().__init__()
        self.large = nn.Conv1d(Cin, Cout, kL, padding=kL//2, groups=Cin, bias=bias)
        self.small = nn.Conv1d(Cin, Cout, kS, padding=kS//2, groups=Cin, bias=False)
    def forward(self, x):
        return self.small(x) + self.large(x)

class FeedForward(nn.Module):
    def __init__(self, cfg, H: int):
        super().__init__()
        hidden = int(H * cfg.ffn_ratio)
        self.conv1 = nn.Conv1d(H, hidden, 1)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(cfg.drop_ffn)
        self.conv2 = nn.Conv1d(hidden, H, 1)

    def forward(self, x):
        B, M, H, T = x.shape
        x = x.reshape(B * M, H, T)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x.reshape(B, M, H, T)

# --------------------------------------------------------------------------- #
# 4. Simplified SACA Implementation
# --------------------------------------------------------------------------- #
def _get_saca_output_dim(mode: int, enc_in: int, n_heads: int) -> int:
    """Get output dimension based on SACA parameter mode

    Args:
        mode: 0=not used, 1=[enc_in,H], 2=[enc_in,1], 3=[1,H], 4=[1,1]
        enc_in: number of input channels
        n_heads: number of attention heads
    """
    if mode == 0:
        return 0  # Not used
    elif mode == 1:
        return enc_in * n_heads  # [enc_in, H]
    elif mode == 2:
        return enc_in  # [enc_in, 1]
    elif mode == 3:
        return n_heads  # [1, H]
    elif mode == 4:
        return 1  # [1, 1]
    else:
        raise ValueError(f"Invalid SACA mode: {mode}. Must be 0-4.")

def _reshape_saca_param(param: torch.Tensor, mode: int, enc_in: int, n_heads: int) -> torch.Tensor:
    """Reshape SACA parameter to broadcasting-ready dimensions

    Args:
        param: parameter tensor [B, output_dim]
        mode: SACA parameter mode (0-4)
        enc_in: number of input channels
        n_heads: number of attention heads

    Returns:
        reshaped tensor ready for broadcasting with attention tensors [B, H, M, T]
    """
    if param is None or mode == 0:
        # Return ones tensor with minimal broadcasting dimensions
        B = param.size(0) if param is not None else 1
        return torch.ones(B, 1, 1, 1, device=param.device, dtype=param.dtype)

    B = param.size(0)

    if mode == 1:  # [enc_in, H] → [B, H, M, 1]
        return param.view(B, enc_in, n_heads).permute(0, 2, 1).unsqueeze(-1)
    elif mode == 2:  # [enc_in, 1] → [B, 1, M, 1] (broadcast across heads)
        return param.view(B, enc_in, 1).permute(0, 2, 1).unsqueeze(-1)
    elif mode == 3:  # [1, H] → [B, H, 1, 1] (broadcast across enc_in)
        return param.view(B, 1, n_heads).permute(0, 2, 1).unsqueeze(-1)
    elif mode == 4:  # [1, 1] → [B, 1, 1, 1] (broadcast across both)
        return param.view(B, 1, 1, 1)
    else:
        raise ValueError(f"Invalid SACA mode: {mode}")

class SACAReparameterization(nn.Module):
    """Simplified SACA learner for a single parameter with Dropout"""
    def __init__(self, enc_in: int, n_heads: int, mode: int, hidden_dim: int = 128,
                 drop_rate: float = 0.0, param_type: str = 'sig'):
        super().__init__()
        self.mode = mode
        self.enc_in = enc_in
        self.n_heads = n_heads

        if mode == 0:
            self.net = None
            return

        output_dim = _get_saca_output_dim(mode, enc_in, n_heads)

        # Set bias for final layer based on parameter type
        final_bias = True if param_type == 'sig' else False

        # Build the sequential model
        self.net = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, output_dim, bias=final_bias),
        )

    def forward(self, stats: torch.Tensor, param_type: str = 'sig') -> Optional[torch.Tensor]:
        """
        Args:
            stats: [B, 1, enc_in] input statistics
            param_type: 'sig' for scale parameters, 'mu' for bias parameters
        Returns:
            parameter tensor shaped for broadcasting with [B, H, M, T]
        """
        if self.net is None or self.mode == 0:
            return None

        raw_output = self.net(stats.squeeze(1))  # [B, output_dim]
        reshaped = _reshape_saca_param(raw_output, self.mode, self.enc_in, self.n_heads)
        return reshaped.to(stats.dtype)


class SACAModule(nn.Module):
    """Simplified SACA module with unified dropout"""
    def __init__(self, cfg, enc_in: int, n_heads: int):
        super().__init__()
        self.enabled = getattr(cfg, 'SACA', False)
        self.enc_in = enc_in
        self.n_heads = n_heads

        if not self.enabled:
            return

        # Unified hidden dimension calculation using ratio
        # hidden_dim = sqrt(enc_in) * saca_hidden_dim_ratio
        saca_hidden_dim_ratio = getattr(cfg, 'saca_hidden_dim_ratio', 8)  # default ratio of 8
        hidden_dim = int(math.sqrt(enc_in) * saca_hidden_dim_ratio)

        # Ensure hidden_dim is at least 1 if ratio is non-zero
        if saca_hidden_dim_ratio > 0 and hidden_dim < 1:
            hidden_dim = 1

        # Unified dropout rate for all SACA parameters
        drop_saca = getattr(cfg, 'drop_saca', 0.0)

        # Create learners for each parameter with unified hidden_dim
        param_configs = {
            'sig_q': (getattr(cfg, 'sig_q_mode', 1), hidden_dim, 'sig'),
            'sig_k': (getattr(cfg, 'sig_k_mode', 1), hidden_dim, 'sig'),
            'sig_v': (getattr(cfg, 'sig_v_mode', 1), hidden_dim, 'sig'),
            'mu_q': (getattr(cfg, 'mu_q_mode', 1), hidden_dim, 'mu'),
            'mu_k': (getattr(cfg, 'mu_k_mode', 1), hidden_dim, 'mu'),
            'mu_v': (getattr(cfg, 'mu_v_mode', 1), hidden_dim, 'mu'),
        }

        self.learners = nn.ModuleDict()
        for param_name, (mode, hidden_dim, param_type) in param_configs.items():
            # Only create learner if mode > 0 AND hidden_dim > 0
            if mode > 0 and hidden_dim > 0:
                self.learners[param_name] = SACAReparameterization(
                    enc_in, n_heads, mode, hidden_dim, drop_saca, param_type=param_type
                )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                std_stats: Optional[torch.Tensor] = None,
                mean_stats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply SACA transformations

        Args:
            q, k, v: attention tensors [B, H, M, T]
            std_stats: standard deviation statistics [B, 1, enc_in]
            mean_stats: mean statistics [B, 1, enc_in]

        Returns:
            Transformed (q, k, v) tensors
        """
        if not self.enabled or len(self.learners) == 0:
            return q, k, v

        # Apply SACA transformations
        if std_stats is not None:
            # Apply scale parameters (sig_*) with softplus activation and clamping
            if 'sig_q' in self.learners:
                scale_param = F.softplus(self.learners['sig_q'](std_stats, 'sig')).clamp(min=0.01)
                q = scale_param * q
            if 'sig_k' in self.learners:
                scale_param = F.softplus(self.learners['sig_k'](std_stats, 'sig')).clamp(min=0.01)
                k = scale_param * k
            if 'sig_v' in self.learners:
                scale_param = F.softplus(self.learners['sig_v'](std_stats, 'sig')).clamp(min=0.01)
                v = scale_param * v

        if mean_stats is not None:
            # Apply bias parameters (mu_*)
            if 'mu_q' in self.learners:
                bias_param = self.learners['mu_q'](mean_stats, 'mu')
                q = q + bias_param
            if 'mu_k' in self.learners:
                bias_param = self.learners['mu_k'](mean_stats, 'mu')
                k = k + bias_param
            if 'mu_v' in self.learners:
                bias_param = self.learners['mu_v'](mean_stats, 'mu')
                v = v + bias_param

        return q, k, v

# --------------------------------------------------------------------------- #
# 5. Attention with PyTorch SDPA
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg, T: int, kL: int, kS: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_heads
        M = cfg.enc_in
        self.T = T
        self.M = M

        # Always use shared projections for Q, K, V
        self.q_proj = DepthwiseMix(H, H, kL, kS, cfg.qkv_bias)
        self.k_proj = DepthwiseMix(H, H, kL, kS, cfg.qkv_bias)
        self.v_proj = DepthwiseMix(H, H, kL, kS, cfg.qkv_bias)

        self.proj = nn.Conv1d(H, H, 1)
        self.drop_proj = nn.Dropout(cfg.drop_proj)

        # SACA module
        self.saca = SACAModule(cfg, cfg.enc_in, cfg.n_heads)

    def forward(self, x, std_stats=None, mean_stats=None):
        B, M, H, T = x.shape

        # Compute Q, K, V with shared projections (simplified)
        x_reshaped = x.reshape(B*M, H, T)
        q = self.q_proj(x_reshaped).reshape(B, M, H, T).permute(0, 2, 1, 3)  # [B, H, M, T]
        k = self.k_proj(x_reshaped).reshape(B, M, H, T).permute(0, 2, 1, 3)  # [B, H, M, T]
        v = self.v_proj(x_reshaped).reshape(B, M, H, T).permute(0, 2, 1, 3)  # [B, H, M, T]

        if self.cfg.cosine_attention:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # Apply SACA transformations
        q, k, v = self.saca(q, k, v, std_stats, mean_stats)

        # Scaled Dot-Product Attention (PyTorch native SDPA)
        # Automatically selects optimal backend: Flash, Memory-Efficient, or Math
        # Tensor format [B, H, M, T] matches SDPA's [B, num_heads, seq_len, head_dim]
        scale = 1.0 / math.sqrt(T)
        dropout_p = getattr(self.cfg, 'drop_attn', 0.0) if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=False
        )

        out = self.proj(out.transpose(1,2).reshape(B*M, H, T))
        out = self.drop_proj(out.reshape(B, M, H, T))

        return out

# --------------------------------------------------------------------------- #
# 6. Simple DropPath Implementation
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x

        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

# --------------------------------------------------------------------------- #
# 7. T1Block with Simplified SACA
# --------------------------------------------------------------------------- #
class T1Block(nn.Module):
    def __init__(self, cfg, T: int, kL: int, kS: int):
        super().__init__()
        M, H = cfg.enc_in, cfg.n_heads
        self.cfg = cfg

        self.attn = Attention(cfg, T, kL, kS)
        self.ffn = FeedForward(cfg, H)
        self.dp1 = DropPath(cfg.drop_path)
        self.dp2 = DropPath(cfg.drop_path)

        self.norm1 = nn.LayerNorm((H, T), eps=1e-5)
        self.norm2 = nn.LayerNorm((H, T), eps=1e-5)
        self.scale1 = nn.Parameter(torch.ones(1, 1, 1, 1) * 1e-6)
        self.scale2 = nn.Parameter(torch.ones(1, 1, 1, 1) * 1e-6)

    def forward(self, x, std_stats=None, mean_stats=None):
        # First attention block with SACA
        attn_out = self.attn(x, std_stats, mean_stats)
        y1 = self.norm1(attn_out)
        y1_scaled = self.scale1 * y1
        y1_dropped = self.dp1(y1_scaled)
        x = x + y1_dropped

        # Second FFN block
        y2 = self.norm2(self.ffn(x))
        y2_scaled = self.scale2 * y2
        y2_dropped = self.dp2(y2_scaled)
        x = x + y2_dropped

        return x

# --------------------------------------------------------------------------- #
# 8. Stage & Downsample
# --------------------------------------------------------------------------- #
class DownSample(nn.Module):
    def __init__(self, k: int, s: int, C: int):
        super().__init__()
        self.k, self.s = k, s
        self.dw = nn.Conv1d(C, C, k, s, groups=C)

    def forward(self, x, *args):
        B, M, C, T = x.shape
        # Calculate padding needed for ceil division output
        out_len = ceil_div(T, self.s)
        total_len_needed = (out_len - 1) * self.s + self.k
        pad = max(0, total_len_needed - T)
        if pad:
            x = torch.cat([x[..., -1:].repeat(1,1,1,pad), x], -1)
        x = self.dw(x.reshape(B*M, C, -1))
        return (x.reshape(B, M, C, x.size(-1)),) + args

class T1Stage(nn.Module):
    def __init__(self, cfg, n_blk: int, T: int, last: bool, kL: int, kS: int):
        super().__init__()
        self.blocks = nn.ModuleList()

        # Create blocks with SACA configuration
        for i in range(n_blk):
            # Apply SACA_first_only logic
            block_cfg = cfg
            if getattr(cfg, 'SACA_first_only', False) and getattr(cfg, 'SACA', False) and i > 0:
                block_cfg = copy.copy(cfg)
                block_cfg.SACA = False

            self.blocks.append(T1Block(block_cfg, T, kL, kS))

        # Downsampling layer
        self.down = None
        if not last:
            self.down = DownSample(cfg.downsample_ratio, cfg.downsample_ratio, cfg.n_heads)

    def forward(self, x, std_stats=None, mean_stats=None):
        # Pass through all blocks
        for blk in self.blocks:
            x = blk(x, std_stats, mean_stats)

        # Apply downsampling if present
        if self.down:
            x, _, _ = self.down(x, std_stats, mean_stats)

        return x

# --------------------------------------------------------------------------- #
# 9. Heads
# --------------------------------------------------------------------------- #
class ForecastHead(nn.Module):
    def __init__(self, cfg, T_out: int):
        super().__init__()
        proj_dim_cfg = getattr(cfg, 'projection_head_dim', 0)

        if proj_dim_cfg > 0:
            self.proj = nn.Linear(cfg.n_heads, proj_dim_cfg)
            proj_dim = proj_dim_cfg
        else:
            self.proj = nn.Identity()
            proj_dim = cfg.n_heads

        self.fc = nn.Linear(proj_dim * T_out, cfg.pred_len)
        self.dp = nn.Dropout(getattr(cfg, 'drop_head', 0.0))

    def forward(self, x):
        B, M, H, T = x.shape
        x = x.reshape(B * M, H, T).transpose(1, 2)
        x = self.proj(x)
        x = x.flatten(start_dim=1)
        out = self.fc(self.dp(x))
        return out.reshape(B, M, -1)

class PixelShuffle1D(nn.Module):
    def __init__(self, r: int):
        super().__init__(); self.r = r
    def forward(self, x):
        B, C, L = x.shape
        assert C % self.r == 0
        out = x.reshape(B, C//self.r, self.r, L).permute(0,1,3,2)
        return out.reshape(B, C//self.r, L*self.r)

class ReconHead(nn.Module):
    def __init__(self, cfg, T_out: int):
        super().__init__()
        self.pred_len = (cfg.seq_len + cfg.pred_len) if cfg.task_name.endswith("forecast") else cfg.pred_len

        # Anomaly detection bottleneck structure (default: 0 = disabled)
        bottle_neck_n_heads = getattr(cfg, 'bottle_neck_n_heads', 0)
        if bottle_neck_n_heads > 0 and cfg.task_name == 'anomaly_detection':
            self.use_bottleneck = True
            self.bottleneck_dim = bottle_neck_n_heads
            self.proj_down = nn.Conv1d(cfg.n_heads, self.bottleneck_dim, 1)
            self.proj_up = nn.Conv1d(self.bottleneck_dim, cfg.n_heads, 1)
        else:
            self.use_bottleneck = False

        # Always use ceil_div for upsampling factor
        self.up = ceil_div(self.pred_len, T_out)

        # Find adjusted channels (smallest multiple of up >= n_heads)
        self.adjusted_channels = ((cfg.n_heads + self.up - 1) // self.up) * self.up

        # 1x1 conv to adjust channels before pixel shuffle
        self.channel_adjust = nn.Conv1d(cfg.n_heads, self.adjusted_channels, 1)

        # Pixel shuffle
        self.ps = PixelShuffle1D(self.up)

        # Actual length after pixel shuffle
        self.ps_output_len = T_out * self.up

        # Output channels after pixel shuffle
        outC = self.adjusted_channels // self.up

        # Final projection
        self.proj = nn.Linear(outC, 1) if cfg.head_params_shared else nn.Conv1d(self.pred_len, self.pred_len, kernel_size=outC, groups=self.pred_len)
        self.dp = nn.Dropout(cfg.drop_head)

    def forward(self, x):
        B, M, H, T = x.shape

        # Apply bottleneck if enabled (for anomaly detection)
        if self.use_bottleneck:
            x = x.reshape(B * M, H, T)
            x = self.proj_down(x)
            x = self.proj_up(x)
            x = x.reshape(B, M, H, T)

        # Adjust channels to be multiple of upsampling factor
        x = x.reshape(B * M, H, T)
        x = self.channel_adjust(x)  # H -> adjusted_channels

        # Pixel shuffle
        y = self.ps(x.reshape(B, M * self.adjusted_channels, T)).reshape(B * M, self.adjusted_channels // self.up, self.ps_output_len)

        # Center crop if needed
        if self.ps_output_len > self.pred_len:
            crop_start = (self.ps_output_len - self.pred_len) // 2
            y = y[..., crop_start:crop_start + self.pred_len]

        # Final projection
        y = y.transpose(1, 2)
        return self.dp(self.proj(y)).reshape(B, M, self.pred_len)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ClassHead(nn.Module):
    def __init__(self, cfg, T_out: int):
        super().__init__()
        input_features_to_linear = cfg.enc_in * cfg.n_heads
        proj_dim = cfg.projection_head_dim

        self.projection = nn.Linear(input_features_to_linear, proj_dim)
        self.activation = Swish()
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(cfg.drop_head)
        self.linear = nn.Linear(proj_dim, cfg.num_class)

        self.final_pool_type = getattr(cfg, 'final_pool_type', 'gap').lower()
        if self.final_pool_type == 'gap':
            self.final_pool = lambda x: x.mean(dim=1)
        elif self.final_pool_type == 'maxpool':
            self.final_pool = lambda x: torch.max(x, dim=1).values
        else:
            raise ValueError(f"Unsupported final pooling type: {self.final_pool_type}. Choose 'gap' or 'maxpool'.")

    def forward(self, x):
        B, M, H, T = x.shape
        x = x.reshape(B, M * H, T)
        x = self.projection(x.transpose(1, 2))
        x = self.activation(x)
        x = self.final_pool(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# --------------------------------------------------------------------------- #
# 10. Model with Mask-Aware Imputation
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg  = cfg
        self.task = cfg.task_name
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.label_len = getattr(cfg, 'label_len', 48)

        self.stem_pad = FrontPadding(cfg.patch_size, cfg.patch_stride)

        # Determine input channels based on task and mask embedding setting
        if self.task == 'imputation' and getattr(cfg, 'imputation_use_mask_embedding', False):
            input_channels = 2
        else:
            input_channels = 1

        self.stem = nn.Conv1d(input_channels, cfg.n_heads, cfg.patch_size, cfg.patch_stride)

        # Compute T after stem with front padding
        T_after_stem = compute_out_len(cfg.seq_len, cfg.patch_size, cfg.patch_stride)
        if cfg.positional_encoding:
            self.pos = nn.Parameter(torch.randn(1, cfg.enc_in, cfg.n_heads, T_after_stem) * .02)

        # Build stages
        stages, curT = [], T_after_stem
        for i, n in enumerate(cfg.n_blocks):
            kL = pick(cfg.kernel_size_large, i)
            kS = pick(cfg.kernel_size_small, i)
            is_last_stage = (i == len(cfg.n_blocks) - 1)

            stages.append(T1Stage(cfg, n, curT, is_last_stage, kL, kS))

            if not is_last_stage:
                curT = compute_out_len(curT, cfg.downsample_ratio, cfg.downsample_ratio)

        self.stages = nn.ModuleList(stages)

        if self.task == 'classification':
            self.head = ClassHead(cfg, curT)
        elif getattr(cfg, 'use_head_reconstruction', True):
            self.head = ReconHead(cfg, curT)
        else:
            self.head = ForecastHead(cfg, curT)

    def _embed(self, x, mask=None):
        B, T, M = x.shape

        # Apply padding to input
        x_padded = self.stem_pad(x.permute(0, 2, 1))  # [B, M, T_padded]

        # Check if we should use mask embedding for imputation
        use_mask = (self.task == 'imputation' and
                   getattr(self.cfg, 'imputation_use_mask_embedding', False) and
                   mask is not None)

        if use_mask:
            mask = mask.to(x.dtype) / T
            mask_padded = self.stem_pad(mask.permute(0, 2, 1))
            x_padded = x_padded.reshape(B * M, 1, -1)
            mask_padded = mask_padded.reshape(B * M, 1, -1)
            x_input = torch.cat([x_padded, mask_padded], dim=1)
        else:
            x_input = x_padded.reshape(B * M, 1, -1)

        x_stemmed = self.stem(x_input)
        x_out = x_stemmed.reshape(B, M, self.cfg.n_heads, -1)

        if self.cfg.positional_encoding:
            x_out = x_out + self.pos

        return x_out

    def forward_features(self, x, std_stats=None, mean_stats=None, mask=None):
        x = self._embed(x, mask=mask)
        for stage in self.stages:
            x = stage(x, std_stats, mean_stats)
        return x

    def _normalize_input(self, x, mask=None, task_type='standard'):
        if task_type == 'imputation' and mask is not None:
            n_observed = torch.sum(mask == 1, dim=1, keepdim=True)  # [B, 1, M]

            is_zero_obs = (n_observed == 0)
            is_one_obs = (n_observed == 1)
            is_normal = ~(is_zero_obs | is_one_obs)

            mean_enc = torch.zeros_like(n_observed, dtype=x.dtype)
            std_enc = torch.ones_like(n_observed, dtype=x.dtype)

            if is_one_obs.any():
                mean_one = torch.sum(x * mask, dim=1, keepdim=True)
                mean_enc = torch.where(is_one_obs, mean_one, mean_enc)

            if is_normal.any():
                safe_n = torch.where(is_normal, n_observed, torch.ones_like(n_observed))
                mean_normal = torch.sum(x * mask, dim=1, keepdim=True) / safe_n
                mean_enc = torch.where(is_normal, mean_normal, mean_enc)

                x_centered = (x - mean_enc) * mask
                variance = torch.sum(x_centered * x_centered, dim=1, keepdim=True) / torch.maximum(safe_n - 1, torch.ones_like(safe_n))
                std_normal = torch.sqrt(variance + 1e-5)
                std_enc = torch.where(is_normal, std_normal, std_enc)

            mean_enc = mean_enc.detach()
            std_enc = std_enc.detach()
            x_norm = (x - mean_enc) * mask / std_enc
        else:
            mean_enc = x.mean(1, keepdim=True).detach()
            x_centered = x - mean_enc
            std_enc = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = x_centered / std_enc

        return x_norm, mean_enc, std_enc

    def _forward_with_saca(self, x_norm, mean_enc, std_enc, mask=None):
        if getattr(self.cfg, 'SACA', False):
            features = self.forward_features(x_norm, std_enc, mean_enc, mask=mask)
        else:
            features = self.forward_features(x_norm, mask=mask)
        return features

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if getattr(self.cfg, 'use_SAN', False) or getattr(self.cfg, 'use_FAN', False):
            mean_enc, std_enc = None, None
        else:
            x_enc, mean_enc, std_enc = self._normalize_input(x_enc, task_type='standard')

        features = self._forward_with_saca(x_enc, mean_enc, std_enc, mask=None)
        y = self.head(features)

        if getattr(self.cfg, 'use_SAN', False) or getattr(self.cfg, 'use_FAN', False):
            y = y.permute(0,2,1)
        else:
            y = y.permute(0,2,1) * std_enc + mean_enc

        return y

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_norm, mean_enc, std_enc = self._normalize_input(x_enc, mask, task_type='imputation')
        features = self._forward_with_saca(x_norm, mean_enc, std_enc, mask=mask)
        y = self.head(features).permute(0,2,1)
        y = y * std_enc + mean_enc
        return y

    def anomaly_detection(self, x):
        x_norm, mean_enc, std_enc = self._normalize_input(x, task_type='standard')
        features = self._forward_with_saca(x_norm, mean_enc, std_enc, mask=None)
        y = self.head(features).permute(0,2,1)
        y = y * std_enc + mean_enc
        return y

    def classification(self, x):
        x_norm, mean_enc, std_enc = self._normalize_input(x, task_type='standard')
        features = self._forward_with_saca(x_norm, mean_enc, std_enc, mask=None)
        return self.head(features)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task == 'classification':
            return self.classification(x_enc)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
