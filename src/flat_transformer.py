import os
import re
import math
import random
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict

def get_video_id_from_pt(filename):
    return filename.split("_clip")[0]

def get_clip_index(filename):
    match = re.search(r"_clip-(\d+)", filename)
    if match is None:
        raise ValueError(f"Could not parse clip index from filename: {filename}")
    return int(match.group(1))

def group_feature_files_by_video(feature_dir, expected_clips=7):
    groups = defaultdict(list)

    for fname in os.listdir(feature_dir):
        if fname.endswith(".pt"):
            vid = get_video_id_from_pt(fname)
            groups[vid].append(os.path.join(feature_dir, fname))

    for vid in groups:
        groups[vid] = sorted(groups[vid], key=lambda p: get_clip_index(os.path.basename(p)))

    bad_videos = [vid for vid, files in groups.items() if len(files) != expected_clips]
    if bad_videos:
        raise ValueError(
            f"Expected {expected_clips} clips per video, but found mismatches for these videos: {bad_videos[:10]}"
        )

    return groups

def split_videos_no_leakage(groups, train_ratio=0.7, val_ratio=0.1, seed=42):
    vids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(vids)

    n = len(vids)
    train_idx = int(n*train_ratio)
    val_idx = int(n*(train_ratio + val_ratio))

    train_vids = vids[:train_idx]
    val_vids = vids[train_idx:val_idx]
    test_vids = vids[val_idx:]
    return train_vids, val_vids, test_vids

# Video-level dataset
# Each item = one full video tensor: [7, 12, 512]

class VideoFeatureDataset(Dataset):
    def __init__(self, groups, video_ids, expected_clips=7, expected_frames=12, input_dim=512):
        self.groups = groups
        self.video_ids = video_ids
        self.expected_clips = expected_clips
        self.expected_frames = expected_frames
        self.input_dim = input_dim

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        clip_files = self.groups[vid]

        if len(clip_files) != self.expected_clips:
            raise ValueError(f"Video {vid} has {len(clip_files)} clips, expected {self.expected_clips}")

        clip_tensors = []
        labels = []

        for f in clip_files:
            data = torch.load(f, map_location="cpu")
            feats = data["features"]   # expected [12, 512]
            label = data["label"]

            if feats.shape != (self.expected_frames, self.input_dim):
                raise ValueError(
                    f"Clip {os.path.basename(f)} has shape {tuple(feats.shape)}, "
                    f"expected {(self.expected_frames, self.input_dim)}"
                )

            clip_tensors.append(feats)
            labels.append(label)

        # all clips from the same video should have the same label
        if len(set(labels)) != 1:
            raise ValueError(f"Video {vid} has inconsistent clip labels: {labels}")

        x = torch.stack(clip_tensors, dim=0)   # [7, 12, 512]
        y = labels[0]

        return x.float(), torch.tensor(y, dtype=torch.long)

class SinCosPositionalEncoding(nn.Module):
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

class FlatTemporalTransformer(nn.Module):
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
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.seq_len = num_clips * frames_per_clip   # 84

        # 512 -> 256 compression
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # +1 because of CLS
        self.pos_encoding = SinCosPositionalEncoding(d_model=d_model, max_len=self.seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

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
        """
        x: [B, 7, 12, 512]
        """
        B, C, F, D = x.shape

        if C != self.num_clips or F != self.frames_per_clip or D != self.input_dim:
            raise ValueError(
                f"Expected input shape [B, {self.num_clips}, {self.frames_per_clip}, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )

        # project feature dimension: [B, 7, 12, 512] -> [B, 7, 12, 256]
        x = self.input_proj(x)
        # flatten clip and frame dimensions: [B, 7, 12, 256] -> [B, 84, 256]
        x = x.reshape(B, self.seq_len, self.d_model)
        # add global CLS token: [B, 1, 256] + [B, 84, 256] -> [B, 85, 256]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # positional encoding
        x = self.pos_encoding(x)

        x = self.encoder(x)

        cls_out = x[:, 0]
        logits = self.classifier(cls_out)

        return logits