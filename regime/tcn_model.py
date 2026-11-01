# regime/tcn_model.py
from __future__ import annotations
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _ = x.shape
        y = x.mean(dim=-1)
        y = self.fc(y).view(b, c, 1)
        return x * y

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        se_reduction: int = 16,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.ln1 = nn.LayerNorm(n_outputs)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.ln2 = nn.LayerNorm(n_outputs)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.se = SqueezeExcitation1d(n_outputs, se_reduction) if se_reduction > 0 else nn.Identity()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        self.act = nn.GELU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = out.transpose(1, 2)
        out = self.ln1(out)
        out = out.transpose(1, 2)
        out = self.gelu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)
        out = self.gelu2(out)
        out = self.dropout2(out)

        out = self.se(out)
        out = self.stochastic_depth(out)
        out = out + residual
        return self.act(out)

class TCNEncoder(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        se_reduction: int = 16,
        stochastic_depth_prob: float = 0.1,
        max_length: int = 2000,
    ):
        super().__init__()
        self.max_length = max_length
        self.pos_enc = nn.Parameter(torch.randn(1, num_inputs, max_length) * 0.02)
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout=dropout,
                    se_reduction=se_reduction,
                    stochastic_depth_prob=stochastic_depth_prob,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        seq_len = x.shape[-1]
        pos = self.pos_enc[:, :, :seq_len]
        x = x + pos
        return self.network(x)

class HorizonAttentionPool(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, seq):
        key = seq.permute(2, 0, 1)   # (L, B, C)
        batch_size = key.shape[1]
        query = self.query_token.expand(1, batch_size, -1)
        attn_out, _ = self.attn(query, key, key)
        return attn_out.squeeze(0)   # (B, C)

class MultiHorizonRegimeTCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_horizons: int,
        num_classes: int = 3,
        num_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.3,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        se_reduction: int = 16,
        stochastic_depth_prob: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.encoder = TCNEncoder(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            se_reduction=se_reduction,
            stochastic_depth_prob=stochastic_depth_prob,
            max_length=max_seq_len,
        )
        d_model = num_channels[-1]
        self.pools = nn.ModuleList([
            HorizonAttentionPool(d_model, attention_heads, attention_dropout)
            for _ in range(num_horizons)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_horizons)
        ])

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)                 # (B, C, L)
        enc = self.encoder(x)                 # (B, C, L)
        outputs = []
        for pool, head in zip(self.pools, self.heads):
            context = pool(enc)               # (B, C)
            logits = head(context)            # (B, num_classes)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)    # (B, num_horizons, num_classes)