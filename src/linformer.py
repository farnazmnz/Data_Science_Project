import torch
import torch.nn as nn
import math

class LinformerSinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LinformerSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, seq_len, k_proj=16, dropout=0.1):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.seq_len = seq_len
        self.k_proj = k_proj

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj_in = nn.Linear(d_model, d_model)
        self.v_proj_in = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.E = nn.Parameter(torch.randn(nhead, k_proj, seq_len) * 0.02)
        self.F = nn.Parameter(torch.randn(nhead, k_proj, seq_len) * 0.02)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        if N != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {N}")

        q = self.q_proj(x)
        k = self.k_proj_in(x)
        v = self.v_proj_in(x)

        q = q.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        k_lin = torch.einsum("bhnd,hkn->bhkd", k, self.E)
        v_lin = torch.einsum("bhnd,hkn->bhkd", v, self.F)

        scores = torch.matmul(q, k_lin.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v_lin)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return out


class LinformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, seq_len, k_proj=16, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinformerSelfAttention(
            d_model=d_model,
            nhead=nhead,
            seq_len=seq_len,
            k_proj=k_proj,
            dropout=dropout
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class LinformerTemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=512,
        d_model=256,
        num_classes=8,
        num_clips=7,
        frames_per_clip=12,
        nhead=4,
        num_layers=2,
        ff_dim=512,
        k_proj=16,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.seq_len = num_clips * frames_per_clip

        # 512 -> 256 compression
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # global CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_encoding = LinformerSinCosPositionalEncoding(
            d_model=d_model,
            max_len=self.seq_len + 1
        )

        self.layers = nn.ModuleList([
            LinformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                ff_dim=ff_dim,
                seq_len=self.seq_len + 1,
                k_proj=k_proj,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [B, 7, 12, 512]
        B, C, F, D = x.shape

        if C != self.num_clips or F != self.frames_per_clip or D != self.input_dim:
            raise ValueError(
                f"Expected input shape [B, {self.num_clips}, {self.frames_per_clip}, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )

        # [B, 7, 12, 512] -> [B, 7, 12, 256]
        x = self.input_proj(x)

        # [B, 7, 12, 256] -> [B, 84, 256]
        x = x.reshape(B, self.seq_len, self.d_model)

        # CLS: [B, 85, 256]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # sin/cos positional encoding
        x = self.pos_encoding(x)

        # Linformer encoder stack
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # global CLS output
        cls_out = x[:, 0]
        logits = self.classifier(cls_out)

        return logits