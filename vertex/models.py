import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


# --- LSTMAttentionClassifier ---
class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
            bidirectional=True
        )
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = x.permute(1, 0, 2).contiguous()
        
        lstm_out, _ = self.lstm(x, hidden)
        lstm_out = lstm_out.permute(1, 0, 2).contiguous()
        
        lstm_out_dropped = self.dropout(lstm_out)
        attn_weights = torch.softmax(self.attention_fc(lstm_out_dropped), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context_vector)

# --- TransformerClassifier ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 3, dropout: float = 0.2, dim_feedforward: int = 256):
        super().__init__()
        self.d_model = d_model
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch_size = src.shape[0]
        
        src_embedded = self.encoder(src) * math.sqrt(self.d_model) 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src_with_cls = torch.cat((cls_tokens, src_embedded), dim=1) 

        src_permuted = src_with_cls.permute(1, 0, 2) 
        src_pos = self.pos_encoder(src_permuted)

        output = self.transformer_encoder(src_pos)
        cls_output = output[0]
        return self.decoder(cls_output)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1, p: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, k, d)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(out_ch, out_ch, k, d)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.dropout(y)
        return y + self.down(x)

class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, channels: list[int] = [64, 128, 128], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = input_size
        dilation = 1
        for ch in channels:
            layers.append(TCNBlock(in_ch, ch, k=kernel_size, d=dilation, p=dropout))
            in_ch = ch
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        h = self.tcn(x)
        h = h.mean(dim=-1)
        return self.head(h)

# --- PatchTST ---
class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        dim_feedforward: int = 256
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_size = input_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.chan_embed = nn.Embedding(num_embeddings=input_size, embedding_dim=d_model)

        self.proj = nn.Linear(input_size * patch_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _sinusoidal_pos_enc(n_pos: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(n_pos, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(n_pos, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        _, S, _ = x.shape
        if S < self.patch_len:
            pad = self.patch_len - S
            x = nn.functional.pad(x, (0, 0, pad, 0))
            S = x.size(1)
        n = 1 + (S - self.patch_len) // self.stride
        patches = []
        for i in range(n):
            s = i * self.stride
            e = s + self.patch_len
            patches.append(x[:, s:e, :].unsqueeze(1))
        return torch.cat(patches, dim=1)

    def _channel_context(self, patches: torch.Tensor) -> torch.Tensor:
        B, Np, _, F = patches.shape
        energy = (patches.pow(2).sum(dim=2).sqrt())
        weights = torch.softmax(energy, dim=-1)

        ch_idx = torch.arange(F, device=patches.device)
        ch_emb = self.chan_embed(ch_idx)
        ch_emb = ch_emb.view(1, 1, F, self.d_model).expand(B, Np, F, self.d_model)

        ctx = (weights.unsqueeze(-1) * ch_emb).sum(dim=2)
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, F = x.shape
        patches = self._patchify(x)
        B, Np, L, F = patches.shape

        tokens = self.proj(patches.reshape(B, Np, L * F)) * math.sqrt(self.d_model)

        ctx = self._channel_context(patches)
        tokens = tokens + ctx

        cls = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        pos = self._sinusoidal_pos_enc(1 + Np, self.d_model, tokens.device)
        tokens = tokens + pos.unsqueeze(0)

        z = self.encoder(tokens)
        z = self.dropout(z[:, 0, :])
        return self.head(z)