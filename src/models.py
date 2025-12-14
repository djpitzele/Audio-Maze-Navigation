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


class WavefrontNet(nn.Module):
    """
    [FINAL 2D ARCHITECTURE]
    Treats the 8-mic array as an 'Image' (8 rows x Time columns).
    Uses 2D Convolutions to detect 'wavefronts' (slanted lines)
    passing across the neighboring microphones.
    """
    def __init__(self, num_classes=4, in_channels=1, dropout=0.5):
        super().__init__()
        
        # Layer 1: Detect Phase Shifts (Slopes)
        # Circular padding handles the Mic 0 <-> Mic 7 connection
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 64), stride=(1, 4), padding=(1, 32))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4)) # Pool only time
        
        # Layer 2: Refine Spatial Features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 16), stride=(1, 2), padding=(1, 8))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4)) # Pool spatial (8->4) & time
        
        # Layer 3: Extract Echo Texture
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 8), stride=(1, 1), padding=(1, 4))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 4)) # Pool spatial (4->2)
        
        # Layer 4: Deep Features
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(2, 4), stride=(1, 1), padding=(0, 2))
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1)) # Global Pool
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x input: [Batch, 8, Time]
        # Reshape to [Batch, 1, 8, Time] for 2D Conv
        x = x.unsqueeze(1)
        
        # Manual Circular Padding for the "Height" dimension (Mics)
        # Pads top with Mic 7 and bottom with Mic 0
        x = F.pad(x, (0, 0, 1, 1), mode='circular') 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class WideFieldNet(nn.Module):
    """
    [Legacy 1D Architecture]
    Wide-Field Acoustic Network designed for Reverberant Environments.
    """
    def __init__(self, num_classes=4, in_channels=8, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.head(self.encoder(x))