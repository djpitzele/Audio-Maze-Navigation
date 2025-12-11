import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, p=2, s=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, stride=s)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class AcousticCNN1D(nn.Module):
    """Lightweight 1D CNN over 8 mic time-series."""
    def __init__(self, num_classes=5, in_channels=8, dropout=0.2):
        super().__init__()
        self.stem = ConvBlock1D(in_channels, 32, k=5, p=2, dropout=dropout)
        self.block1 = ConvBlock1D(32, 64, k=5, p=2, dropout=dropout)
        self.block2 = ConvBlock1D(64, 128, k=3, p=1, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C=8, T)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.head(x)
        return x


class SpatialTemporalAcousticNet(nn.Module):
    """
    Advanced architecture for acoustic navigation with circular mic array.

    Key innovations:
    1. Efficient temporal downsampling (11k -> 128 features)
    2. Spatial processing of 8-mic circular array
    3. Multi-head attention for long-range acoustic dependencies
    4. Residual connections for gradient flow
    """
    def __init__(self, num_classes=5, in_channels=8, dropout=0.3):
        super().__init__()

        # Temporal feature extraction: aggressively downsample long time series
        # 11434 -> 2858 -> 715 -> 179 -> 45 -> 11 -> 1
        self.temporal_encoder = nn.Sequential(
            # Stage 1: Initial compression
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),

            # Stage 2: Feature extraction
            nn.Conv1d(32, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            # Stage 3: More compression
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # Stage 4: Deep features
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # Stage 5: Final compression
            nn.Conv1d(256, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        # Spatial attention: model relationships between mics
        # (mics form a circle, spatial patterns matter for direction)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Temporal attention: capture long-range reverberations
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Residual projection
        self.residual_proj = nn.Linear(256, 256)

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C=8, T=11434)
        B, C, T = x.shape

        # Temporal encoding: extract features from long time series
        x = self.temporal_encoder(x)  # (B, 256, T')

        # Transpose for attention: (B, T', 256)
        x = x.transpose(1, 2)

        # Spatial attention (attend across time dimension)
        x_attn, _ = self.spatial_attention(x, x, x)
        x = x + x_attn  # Residual connection

        # Temporal attention (attend within time)
        x_attn, _ = self.temporal_attention(x, x, x)
        x = x + x_attn  # Residual connection

        # Global pooling
        x = x.mean(dim=1)  # (B, 256)

        # Classification
        x = self.head(x)
        return x


class CompactAcousticNet(nn.Module):
    """
    Efficient baseline: Better than original but lighter than SpatialTemporalAcousticNet.
    Uses strided convolutions for aggressive downsampling.
    """
    def __init__(self, num_classes=5, in_channels=8, dropout=0.3):
        super().__init__()

        # Aggressive temporal downsampling with larger kernels
        self.encoder = nn.Sequential(
            # 11434 -> 2858
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),

            # 2858 -> 715
            nn.Conv1d(32, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            # 715 -> 179
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # 179 -> 45
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # 45 -> 12
            nn.Conv1d(256, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C=8, T=11434)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.head(x)
        return x
