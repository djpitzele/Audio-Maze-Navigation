import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WideFieldNet(nn.Module):
    """
    [FINAL ARCHITECTURE]
    Wide-Field Acoustic Network designed for Reverberant Environments.
    
    Key Innovations:
    1. Large Kernel (64) in Layer 1 to capture long-range Impulse Response (Echoes).
    2. Stride 2 to preserve phase cues better than MaxPool.
    3. LeakyReLU to prevent dead gradients during training.
    """
    def __init__(self, num_classes=4, in_channels=8, dropout=0.5):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: The "Echo Catcher" (64 samples ~= 0.4ms)
            # Stride 2 captures Phase, Kernel 64 captures Context
            nn.Conv1d(in_channels, 64, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            # Layer 2: Texture & Decay
            nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            # Layer 3: Deep Features
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            # Layer 4: Global Context
            nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1), # Squeeze time to 1 vector
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.head(self.encoder(x))


class SpatialTemporalAcousticNet(nn.Module):
    """
    (Legacy) Advanced architecture with attention mechanisms.
    """
    def __init__(self, num_classes=5, in_channels=8, dropout=0.3):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
        )
        self.spatial_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        x_attn, _ = self.spatial_attention(x, x, x)
        x = x + x_attn
        x_attn, _ = self.temporal_attention(x, x, x)
        x = x + x_attn
        x = x.mean(dim=1)
        return self.head(x)


class CompactAcousticNet(nn.Module):
    """
    (Legacy) Efficient baseline using strided convolutions.
    """
    def __init__(self, num_classes=5, in_channels=8, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)