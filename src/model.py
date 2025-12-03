"""
Deep Learning Models for Acoustic Navigation

This module implements CNN architectures for learning navigation policies
from acoustic spectrograms. The models are designed to process multi-channel
spectrogram inputs and output discrete action probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AudioNavCNN(nn.Module):
    """
    Convolutional Neural Network for acoustic-based navigation.

    This architecture processes multi-channel spectrograms (one per microphone)
    and outputs action probabilities. The model uses:
    - 2D convolutions to extract spatial-temporal-frequency features
    - Batch normalization for training stability
    - Dropout for regularization
    - Fully connected layers for action prediction

    Input Shape
    -----------
    (batch_size, num_microphones, freq_bins, time_bins)

    Output Shape
    ------------
    (batch_size, num_actions)

    Parameters
    ----------
    num_microphones : int, optional
        Number of microphones in the array (default: 8)
    num_actions : int, optional
        Number of discrete actions (default: 5 for STOP, UP, DOWN, LEFT, RIGHT)
    input_freq_bins : int, optional
        Number of frequency bins in spectrogram (default: 33)
    input_time_bins : int, optional
        Number of time bins in spectrogram (default: depends on signal length)
    dropout_rate : float, optional
        Dropout probability for regularization (default: 0.3)

    Architecture
    ------------
    The network consists of:
    1. Three convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    2. Global Average Pooling
    3. Two fully connected layers with dropout
    4. Softmax output layer

    Example
    -------
    >>> model = AudioNavCNN(num_microphones=8, num_actions=5)
    >>> x = torch.randn(32, 8, 33, 64)  # Batch of 32 spectrograms
    >>> logits = model(x)
    >>> print(logits.shape)  # (32, 5)
    """

    def __init__(
        self,
        num_microphones: int = 8,
        num_actions: int = 5,
        input_freq_bins: int = 33,
        input_time_bins: Optional[int] = None,  # Can be variable
        dropout_rate: float = 0.3,
    ):
        super(AudioNavCNN, self).__init__()

        self.num_microphones = num_microphones
        self.num_actions = num_actions
        self.dropout_rate = dropout_rate

        # Convolutional Block 1
        # Input: (batch, num_mics, freq, time)
        self.conv1 = nn.Conv2d(
            in_channels=num_microphones,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, num_actions)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram with shape (batch_size, num_microphones, freq_bins, time_bins)

        Returns
        -------
        torch.Tensor
            Action logits with shape (batch_size, num_actions)
            Use softmax or log_softmax for probabilities.
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def predict_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely action for a given input.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram

        Returns
        -------
        torch.Tensor
            Predicted action indices (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def get_action_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram

        Returns
        -------
        torch.Tensor
            Action probabilities with shape (batch_size, num_actions)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for deeper networks.

    This can be used to build more sophisticated architectures if needed.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int, optional
        Stride for convolution (default: 1)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class AudioNavResNet(nn.Module):
    """
    Residual Network architecture for acoustic navigation.

    A deeper alternative to AudioNavCNN using residual connections
    to facilitate training of deeper networks.

    Parameters
    ----------
    num_microphones : int, optional
        Number of microphones in the array (default: 8)
    num_actions : int, optional
        Number of discrete actions (default: 5)
    dropout_rate : float, optional
        Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        num_microphones: int = 8,
        num_actions: int = 5,
        dropout_rate: float = 0.3,
    ):
        super(AudioNavResNet, self).__init__()

        self.num_microphones = num_microphones
        self.num_actions = num_actions

        # Initial convolution
        self.conv1 = nn.Conv2d(num_microphones, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Global pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_actions)

        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """Create a layer of residual blocks."""
        layers = []

        # First block may have stride > 1 for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def predict_action(self, x: torch.Tensor) -> torch.Tensor:
        """Predict action."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def get_action_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
