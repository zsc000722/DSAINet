"""
DSAINet_SNN_EEGStyle.py

Structure-preserving SNN version of DSAINet for EEG time-series.

Design goals:
1. Keep the original DSAINet macro-structure and attention form:
   PatchEmbedding -> dual ConvTime branches -> intra MHA -> inter MHA -> token attention pooling.
2. Do NOT duplicate EEG input into artificial CV-style spike steps.
   The EEG temporal/token axis is used as the LIF multi-step axis.
3. Use layer-level spike conversion: major learnable transforms are followed by LIF.
4. Use SpikingJelly MultiStepLIFNode directly; no fallback implementation.

Input : (B, 1, C, T)
Output: (B, n_classes)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import functional


class TemporalLIF(nn.Module):
    """
    Apply MultiStepLIFNode along an existing temporal dimension.

    Unlike CV-style SNNs, this module does not create/repeat an artificial
    simulation-step axis. Instead, it treats the given EEG temporal/token axis
    as the multi-step axis.

    Examples:
        (B, C, H, T), time_dim=-1 -> (T, B, C, H) -> LIF -> back
        (B, N, E),    time_dim=1  -> (N, B, E)    -> LIF -> back
        (B, E, N),    time_dim=-1 -> (N, B, E)    -> LIF -> back
    """
    def __init__(self, tau: float = 2.0, detach_reset: bool = True, backend: str = "cupy", time_dim: int = -1):
        super().__init__()
        self.time_dim = time_dim
        self.node = MultiStepLIFNode(tau=tau, detach_reset=detach_reset, backend=backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        td = self.time_dim if self.time_dim >= 0 else x.dim() + self.time_dim
        y = torch.movedim(x, td, 0).contiguous()
        y = self.node(y)
        y = torch.movedim(y, 0, td).contiguous()
        return y


class TokenBatchNorm(nn.Module):
    """BatchNorm1d for token tensors with shape (B, N, E)."""
    def __init__(self, emb_size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2).contiguous()


class PatchEmbeddingSNN(nn.Module):
    """
    DSAINet patch embedding with layer-level spike conversion.

    Original main learnable transforms:
        temporal Conv2d -> spatial Conv2d -> temporal Conv2d

    SNN version:
        Conv2d -> BN -> LIF
        Conv2d -> BN -> LIF
        Pool
        Conv2d -> BN -> LIF
        Pool

    The LIF time axis is the EEG time axis T, not an artificial repeated step.
    """
    def __init__(
        self,
        f1: int = 16,
        kernel_size: int = 64,
        D: int = 2,
        pooling_size1: int = 4,
        pooling_size2: int = 8,
        dropout_rate: float = 0.25,
        number_channel: int = 22,
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "cupy",
    ):
        super().__init__()
        f2 = D * f1
        self.f2 = f2

        self.temporal1 = nn.Conv2d(1, f1, (1, kernel_size), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        # self.lif1 = TemporalLIF(tau, detach_reset, backend, time_dim=-1)
        self.lif1 = nn.ELU()

        self.spatial = nn.Conv2d(f1, f2, (number_channel, 1), groups=f1, padding="valid", bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.lif2 = TemporalLIF(tau, detach_reset, backend, time_dim=-1)
        # self.lif2 = nn.ELU()

        self.pool1 = nn.AvgPool2d((1, pooling_size1))
        self.temporal2 = nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.lif3 = TemporalLIF(tau, detach_reset, backend, time_dim=-1)

        self.pool2 = nn.AvgPool2d((1, pooling_size2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,C,T)
        x = self.bn1(self.temporal1(x))
        x = self.lif1(x)
        x = self.bn2(self.spatial(x))
        x = self.pool1(x)
        x = self.lif2(x)
        x = self.bn3(self.temporal2(x))
        x = self.pool2(x)
        x = self.lif3(x)
        return x  # (B,f2,1,N)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding without dropout."""
    def __init__(self, emb_size: int, length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, length, emb_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        return x + self.pe[:, :n, :].to(x.device)


class TokenProjectionSNN(nn.Module):
    """Linear token projection followed by BN, positional encoding, and LIF."""
    def __init__(
        self,
        in_dim: int,
        emb_size: int,
        pos_len: int,
        dropout: float,
        tau: float,
        detach_reset: bool,
        backend: str,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, emb_size) if in_dim != emb_size else nn.Identity()
        self.bn = TokenBatchNorm(emb_size)
        self.pos = PositionalEncoding(emb_size, length=pos_len, dropout=dropout)
        self.lif = TemporalLIF(tau, detach_reset, backend, time_dim=1)
        self.emb_size = emb_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,in_dim)
        x = self.proj(x)
        x = self.bn(x)
        x = x * math.sqrt(self.emb_size)
        x = self.pos(x)
        # x = self.lif(x)
        return x  # (B,N,E), spike state


class ConvTimeLayerSNN(nn.Module):
    """
    Dilated SNN-TCN block for DSAINet temporal branches.

    This replaces the original ConvTime layer with a standard residual TCN-style
    block along the token/time axis N:

        DW dilated Conv1d -> BN -> LIF
        PW expansion Conv1d -> BN -> LIF
        DW dilated Conv1d -> BN -> LIF
        PW projection Conv1d -> BN
        residual scaling + LIF

    Input / output layout is unchanged:
        x: (B, E, N) -> (B, E, N)

    Notes:
    - Depthwise dilated Conv1d keeps temporal modeling SNN-friendly and expands
      receptive field without greatly increasing parameters.
    - Pointwise Conv1d handles channel mixing / FFN-style expansion.
    - The residual scale alpha is initialized to 0, so each block starts close
      to identity and is safer for direct SNN training.
    """
    def __init__(
        self,
        emb_size: int,
        kernel_size: int,
        expansion: int = 4,
        dropout: float = 0.1,
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "cupy",
        dilation: int = 1,
    ):
        super().__init__()
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}.")

        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size - 1) * dilation // 2

        # First depthwise dilated temporal convolution.
        self.dw1 = nn.Conv1d(
            emb_size, emb_size,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=emb_size,
            bias=False,
        )
        self.bn_dw1 = nn.BatchNorm1d(emb_size)
        self.lif_dw1 = TemporalLIF(tau, detach_reset, backend, time_dim=-1)

        d_ff = expansion * emb_size
        # Pointwise expansion, analogous to TCN/FFN channel expansion.
        # Keep grouped pointwise conv when possible to preserve your original
        # lightweight grouped design. Fall back to dense pointwise if needed.
        pw_groups = 4 if emb_size % 4 == 0 and d_ff % 4 == 0 else 1
        self.pw1 = nn.Conv1d(emb_size, d_ff, 1, groups=pw_groups, bias=False)
        self.bn_pw1 = nn.BatchNorm1d(d_ff)
        self.lif_pw1 = TemporalLIF(tau, detach_reset, backend, time_dim=-1)

        # Second depthwise dilated temporal convolution, making the block closer
        # to a standard residual TCN block with two temporal convolutions.
        self.dw2 = nn.Conv1d(
            d_ff, d_ff,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_ff,
            bias=False,
        )
        self.bn_dw2 = nn.BatchNorm1d(d_ff)
        self.lif_dw2 = TemporalLIF(tau, detach_reset, backend, time_dim=-1)

        self.pw2 = nn.Conv1d(d_ff, emb_size, 1, groups=pw_groups, bias=False)
        self.bn_pw2 = nn.BatchNorm1d(emb_size)

        # Keep residual initially close to identity for stable SNN optimization.
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.lif_out = TemporalLIF(tau, detach_reset, backend, time_dim=-1)

    @staticmethod
    def _match_time_length(y: torch.Tensor, target_len: int) -> torch.Tensor:
        """Crop/pad temporal dimension to keep residual addition safe."""
        cur_len = y.shape[-1]
        if cur_len == target_len:
            return y
        if cur_len > target_len:
            start = (cur_len - target_len) // 2
            return y[..., start:start + target_len]
        pad_total = target_len - cur_len
        left = pad_total // 2
        right = pad_total - left
        return F.pad(y, (left, right))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, E, N), spike state on N
        residual = x
        target_len = x.shape[-1]

        y = self.dw1(x)
        y = self._match_time_length(y, target_len)
        y = self.lif_dw1(self.bn_dw1(y))

        y = self.lif_pw1(self.bn_pw1(self.pw1(y)))

        y = self.dw2(y)
        y = self._match_time_length(y, target_len)
        y = self.lif_dw2(self.bn_dw2(y))

        y = self.bn_pw2(self.pw2(y))
        return self.lif_out(residual + self.alpha * y)


class ConvTimeStackSNN(nn.Module):
    """Stack of SNN-TCN blocks.

    The block structure is shared by both coarse and fine branches, keeping the
    two branches symmetric for later FFB6D-style cross-branch fusion. The only
    intended difference is temporal granularity:
        - coarse branch: dilation grows as 1, 2, 4, ... when dilation_base=2;
        - fine branch: dilation stays 1, 1, 1, ... when dilation_base=1.

    The existing kernel_list still controls branch-specific temporal scales.
    """
    def __init__(
        self,
        emb_size: int,
        kernel_list: List[int],
        expansion: int = 4,
        dropout: float = 0.1,
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "cupy",
        dilation_base: int = 2,
    ):
        super().__init__()
        if dilation_base <= 0:
            raise ValueError(f"dilation_base must be positive, got {dilation_base}.")
        self.dilation_base = dilation_base

        self.layers = nn.ModuleList([
            ConvTimeLayerSNN(
                emb_size,
                k,
                expansion=expansion,
                dropout=dropout,
                tau=tau,
                detach_reset=detach_reset,
                backend=backend,
                dilation=dilation_base ** i,
            )
            for i, k in enumerate(kernel_list)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SpikingFFN(nn.Module):
    """Transformer FFN with Linear -> BN -> LIF -> Linear -> BN, then residual LIF outside."""
    def __init__(
        self,
        emb_size: int,
        expansion: int,
        dropout: float,
        tau: float,
        detach_reset: bool,
        backend: str,
    ):
        super().__init__()
        d_ff = expansion * emb_size
        self.fc1 = nn.Linear(emb_size, d_ff)
        self.bn1 = TokenBatchNorm(d_ff)
        self.lif1 = TemporalLIF(tau, detach_reset, backend, time_dim=1)
        self.fc2 = nn.Linear(d_ff, emb_size)
        self.bn2 = TokenBatchNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class IntraAttnBlockSNN(nn.Module):
    """
    Structure-preserving intra-branch block.
    Attention form remains nn.MultiheadAttention(x, x, x).
    """
    def __init__(
        self,
        emb_size: int,
        heads: int,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "cupy",
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.lif_attn = TemporalLIF(tau, detach_reset, backend, time_dim=1)
        self.ffn = SpikingFFN(emb_size, ffn_expansion, dropout, tau, detach_reset, backend)
        self.lif_ffn = TemporalLIF(tau, detach_reset, backend, time_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x)
        x = self.lif_attn(self.norm1(x + attn_out))
        x = self.lif_ffn(self.norm2(x + self.ffn(x)))
        return x


class InterAttnBlockSNN(nn.Module):
    """
    Structure-preserving inter-branch block.
    Attention form remains:
        branch1 <- MHA(Q=x1, K=x2, V=x2)
        branch2 <- MHA(Q=x2, K=x1, V=x1)
    """
    def __init__(
        self,
        emb_size: int,
        heads: int,
        dropout: float = 0.1,
        ffn_expansion: int = 2,
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "cupy",
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, heads, dropout=0.0, batch_first=True)

        self.norm1a = nn.LayerNorm(emb_size)
        self.norm1b = nn.LayerNorm(emb_size)
        self.norm2a = nn.LayerNorm(emb_size)
        self.norm2b = nn.LayerNorm(emb_size)
        self.beta12 = nn.Parameter(torch.tensor(1.0))
        self.beta21 = nn.Parameter(torch.tensor(1.0))

        self.lif_attn1 = TemporalLIF(tau, detach_reset, backend, time_dim=1)
        self.lif_attn2 = TemporalLIF(tau, detach_reset, backend, time_dim=1)
        self.ffn1 = SpikingFFN(emb_size, ffn_expansion, dropout, tau, detach_reset, backend)
        self.ffn2 = SpikingFFN(emb_size, ffn_expansion, dropout, tau, detach_reset, backend)
        self.lif_ffn1 = TemporalLIF(tau, detach_reset, backend, time_dim=1)
        self.lif_ffn2 = TemporalLIF(tau, detach_reset, backend, time_dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        out1, _ = self.mha(x1, x2, x2)
        y1 = self.lif_attn1(self.norm1a(x1 + self.beta12 * out1))

        out2, _ = self.mha(x2, x1, x1)
        y2 = self.lif_attn2(self.norm2a(x2 + self.beta21 * out2))

        y1 = self.lif_ffn1(self.norm1b(y1 + self.ffn1(y1)))
        y2 = self.lif_ffn2(self.norm2b(y2 + self.ffn2(y2)))
        return y1, y2


class DSAINet_SNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        Chans: int,
        Samples: int,
        emb_size: int = 40,
        heads: int = 4,
        attn_depth: int = 1,
        attn_dropout: float = 0.25,
        # patch embedding
        eeg1_f1: int = 16,
        eeg1_kernel_size: int = 64,
        eeg1_D: int = 2,
        eeg1_pooling_size1: int = 4,
        eeg1_pooling_size2: int = 8,
        eeg1_dropout_rate: float = 0.25,
        # branch kernel lists
        branch_1_kernels: Optional[List[int]] = None,
        branch_2_kernels: Optional[List[int]] = None,
        conv_expansion: int = 4,
        conv_dropout: float = 0.25,
        # intra/inter
        intra_ffn_expansion: int = 2,
        inter_ffn_expansion: int = 2,
        # residual controls
        big_residual: bool = True,
        big_residual_learnable: bool = True,
        # classifier
        cls_dropout: float = 0.25,
        # LIF controls
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "cupy",
        reset_state_each_forward: bool = True,
    ):
        super().__init__()
        if branch_1_kernels is None:
            branch_1_kernels = [11, 15]
        if branch_2_kernels is None:
            branch_2_kernels = [3, 7]

        self.emb_size = emb_size
        self.attn_depth = attn_depth
        self.big_residual = big_residual
        self.reset_state_each_forward = reset_state_each_forward

        pos_len = Samples // (eeg1_pooling_size1 * eeg1_pooling_size2)

        self.patch = PatchEmbeddingSNN(
            f1=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            D=eeg1_D,
            pooling_size1=eeg1_pooling_size1,
            pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
            number_channel=Chans,
            tau=tau,
            detach_reset=detach_reset,
            backend=backend,
        )
        f2 = eeg1_f1 * eeg1_D
        self.token_proj = TokenProjectionSNN(f2, emb_size, pos_len, attn_dropout, tau, detach_reset, backend)

        # Coarse branch: keep the same SNN-TCN block structure, but use growing
        # dilation to enlarge temporal receptive field.
        self.branch1 = ConvTimeStackSNN(
            emb_size, branch_1_kernels,
            expansion=conv_expansion,
            dropout=conv_dropout,
            tau=tau,
            detach_reset=detach_reset,
            backend=backend,
            dilation_base=2,
        )
        # Fine branch: keep the same SNN-TCN block structure for symmetry, but
        # fix dilation to 1 so it remains dense/local and preserves fine detail.
        self.branch2 = ConvTimeStackSNN(
            emb_size, branch_2_kernels,
            expansion=conv_expansion,
            dropout=conv_dropout,
            tau=tau,
            detach_reset=detach_reset,
            backend=backend,
            dilation_base=1,
        )

        if big_residual:
            if big_residual_learnable:
                self.alpha1 = nn.Parameter(torch.tensor(1.0))
                self.alpha2 = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer("alpha1", torch.tensor(1.0), persistent=False)
                self.register_buffer("alpha2", torch.tensor(1.0), persistent=False)
            self.lif_big1 = TemporalLIF(tau, detach_reset, backend, time_dim=1)
            self.lif_big2 = TemporalLIF(tau, detach_reset, backend, time_dim=1)

        self.intra_1 = nn.ModuleList([
            IntraAttnBlockSNN(emb_size, heads, attn_dropout, intra_ffn_expansion, tau, detach_reset, backend)
            for _ in range(attn_depth)
        ])
        self.intra_2 = nn.ModuleList([
            IntraAttnBlockSNN(emb_size, heads, attn_dropout, intra_ffn_expansion, tau, detach_reset, backend)
            for _ in range(attn_depth)
        ])
        self.inter = nn.ModuleList([
            InterAttnBlockSNN(emb_size, heads, attn_dropout, inter_ffn_expansion, tau, detach_reset, backend)
            for _ in range(attn_depth)
        ])

        # Original DSAINet token attention pooling is retained as readout.
        self.token_attn = nn.Linear(emb_size, 1)
        self.classifier = nn.Linear(2 * emb_size, n_classes)

    def forward_features(self, x: torch.Tensor):
        # x: (B,1,C,T)
        fmap = self.patch(x)                         # (B,f2,1,N)
        a0 = fmap.squeeze(2).transpose(1, 2)         # (B,N,f2)
        a0 = self.token_proj(a0)                     # (B,N,E), spike state

        z0 = a0.transpose(1, 2)                      # (B,E,N)
        a1 = self.branch1(z0).transpose(1, 2)        # (B,N,E), spike state
        a2 = self.branch2(z0).transpose(1, 2)        # (B,N,E), spike state

        if self.big_residual:
            a1 = self.lif_big1(a1 + self.alpha1 * a0)
            a2 = self.lif_big2(a2 + self.alpha2 * a0)

        for i in range(self.attn_depth):
            a1 = self.intra_1[i](a1)
            a2 = self.intra_2[i](a2)
            a1, a2 = self.inter[i](a1, a2)

        return a1, a2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reset_state_each_forward:
            functional.reset_net(self)

        B = x.shape[0]
        a1, a2 = self.forward_features(x)

        # Original DSAINet token attention pooling retained.
        x = torch.stack([a1, a2], dim=1)             # (B,2,N,E)
        w = torch.softmax(self.token_attn(x).squeeze(-1), dim=2)
        pooled = (x * w.unsqueeze(-1)).sum(dim=2)    # (B,2,E)
        feat = pooled.reshape(B, -1)
        return self.classifier(feat)


# Drop-in alias
DSAINet = DSAINet_SNN
