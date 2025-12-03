"""
PyTorch Dataset and DataLoader utilities for acoustic navigation.

This module provides Dataset classes for loading pre-computed acoustic
simulation data and preparing it for training deep learning models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class AcousticGridDataset(Dataset):
    """
    PyTorch Dataset for acoustic navigation training data.

    This dataset loads pre-computed spectrograms and action labels from
    HDF5 files generated during the data creation phase. The data includes:
    - Spectrograms from k-Wave acoustic simulations
    - Optimal action labels from the A* oracle
    - Metadata (maze configurations, positions, etc.)

    Parameters
    ----------
    hdf5_path : str or Path
        Path to the HDF5 file containing the dataset
    transform : callable, optional
        Optional transform to apply to spectrograms
    normalize : bool, optional
        If True, normalize spectrograms to zero mean and unit variance (default: True)
    cache_in_memory : bool, optional
        If True, load entire dataset into RAM for faster training (default: False)

    HDF5 File Structure
    -------------------
    Expected structure:
    - 'spectrograms': (N, num_mics, freq_bins, time_bins) float32
    - 'actions': (N,) int32
    - 'positions': (N, 2) int32 (optional)
    - 'maze': (height, width) int32 (optional, stored once)

    Example
    -------
    >>> dataset = AcousticGridDataset('data/training_data.h5')
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for spectrograms, actions in dataloader:
    ...     # Train model
    ...     pass
    """

    def __init__(
        self,
        hdf5_path: str,
        transform: Optional[callable] = None,
        normalize: bool = True,
        cache_in_memory: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.transform = transform
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Open file to get metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.num_samples = len(f['actions'])
            self.spectrogram_shape = f['spectrograms'].shape[1:]

            # Compute normalization statistics if needed
            if self.normalize:
                self._compute_normalization_stats(f)

        # Cache data in memory if requested
        if self.cache_in_memory:
            self._load_into_memory()
        else:
            self.cached_spectrograms = None
            self.cached_actions = None

    def _compute_normalization_stats(self, hdf5_file: h5py.File):
        """
        Compute mean and std for normalization.

        For large datasets, this computes statistics in batches to avoid
        loading everything into memory.
        """
        spectrograms = hdf5_file['spectrograms']

        # For small datasets, compute directly
        if self.num_samples < 10000:
            all_data = spectrograms[:]
            self.mean = np.mean(all_data)
            self.std = np.std(all_data)
        else:
            # For large datasets, compute in batches
            batch_size = 1000
            running_sum = 0.0
            running_sum_sq = 0.0
            total_elements = 0

            for i in range(0, self.num_samples, batch_size):
                end_idx = min(i + batch_size, self.num_samples)
                batch = spectrograms[i:end_idx]

                running_sum += np.sum(batch)
                running_sum_sq += np.sum(batch ** 2)
                total_elements += batch.size

            self.mean = running_sum / total_elements
            self.std = np.sqrt(running_sum_sq / total_elements - self.mean ** 2)

        # Avoid division by zero
        if self.std < 1e-8:
            self.std = 1.0

    def _load_into_memory(self):
        """
        Load entire dataset into RAM for faster access.

        Warning: Only use this for datasets that fit comfortably in memory.
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            self.cached_spectrograms = f['spectrograms'][:]
            self.cached_actions = f['actions'][:]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (spectrogram, action) pair
            - spectrogram: (num_mics, freq_bins, time_bins) float32
            - action: scalar int64
        """
        if self.cache_in_memory:
            # Load from cached memory
            spectrogram = self.cached_spectrograms[idx]
            action = self.cached_actions[idx]
        else:
            # Load from HDF5 file
            with h5py.File(self.hdf5_path, 'r') as f:
                spectrogram = f['spectrograms'][idx]
                action = f['actions'][idx]

        # Convert to torch tensors
        spectrogram = torch.from_numpy(spectrogram.astype(np.float32))
        action = torch.tensor(action, dtype=torch.long)

        # Normalize
        if self.normalize:
            spectrogram = (spectrogram - self.mean) / self.std

        # Apply transform
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return spectrogram, action

    def get_maze(self) -> Optional[np.ndarray]:
        """
        Get the maze configuration if stored in the dataset.

        Returns
        -------
        Optional[np.ndarray]
            Maze array or None if not stored
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'maze' in f:
                return f['maze'][:]
        return None

    def get_normalization_stats(self) -> Tuple[float, float]:
        """
        Get normalization statistics.

        Returns
        -------
        Tuple[float, float]
            (mean, std) used for normalization
        """
        if self.normalize:
            return self.mean, self.std
        else:
            return 0.0, 1.0


class AcousticDataModule:
    """
    Data module for managing train/val/test splits and dataloaders.

    This class provides a convenient interface for creating dataloaders
    with proper train/validation/test splits.

    Parameters
    ----------
    train_path : str
        Path to training HDF5 file
    val_path : str, optional
        Path to validation HDF5 file
    test_path : str, optional
        Path to test HDF5 file
    batch_size : int, optional
        Batch size for dataloaders (default: 32)
    num_workers : int, optional
        Number of worker processes for data loading (default: 4)
    normalize : bool, optional
        Whether to normalize data (default: True)
    cache_in_memory : bool, optional
        Whether to cache data in memory (default: False)

    Example
    -------
    >>> data_module = AcousticDataModule(
    ...     train_path='data/train.h5',
    ...     val_path='data/val.h5',
    ...     batch_size=64
    ... )
    >>> train_loader = data_module.train_dataloader()
    >>> val_loader = data_module.val_dataloader()
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        cache_in_memory: bool = False,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        # Create datasets
        self.train_dataset = AcousticGridDataset(
            train_path,
            normalize=normalize,
            cache_in_memory=cache_in_memory
        )

        self.val_dataset = None
        if val_path is not None:
            self.val_dataset = AcousticGridDataset(
                val_path,
                normalize=normalize,
                cache_in_memory=cache_in_memory
            )

        self.test_dataset = None
        if test_path is not None:
            self.test_dataset = AcousticGridDataset(
                test_path,
                normalize=normalize,
                cache_in_memory=cache_in_memory
            )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """
        Create training dataloader.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle training data (default: True)

        Returns
        -------
        DataLoader
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Create validation dataloader.

        Returns
        -------
        Optional[DataLoader]
            Validation dataloader or None if no validation set
        """
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Create test dataloader.

        Returns
        -------
        Optional[DataLoader]
            Test dataloader or None if no test set
        """
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the datasets.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing dataset statistics
        """
        info = {
            'train_size': len(self.train_dataset),
            'spectrogram_shape': self.train_dataset.spectrogram_shape,
            'batch_size': self.batch_size,
        }

        if self.val_dataset is not None:
            info['val_size'] = len(self.val_dataset)

        if self.test_dataset is not None:
            info['test_size'] = len(self.test_dataset)

        if self.normalize:
            mean, std = self.train_dataset.get_normalization_stats()
            info['normalization'] = {'mean': mean, 'std': std}

        return info


def split_hdf5_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42,
):
    """
    Split a single HDF5 dataset into train/val/test sets.

    Parameters
    ----------
    input_path : str
        Path to input HDF5 file
    output_dir : str
        Directory to save split datasets
    train_ratio : float, optional
        Fraction of data for training (default: 0.8)
    val_ratio : float, optional
        Fraction of data for validation (default: 0.1)
    test_ratio : float, optional
        Fraction of data for testing (default: 0.1)
    shuffle : bool, optional
        Whether to shuffle data before splitting (default: True)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)

    Notes
    -----
    The sum of train_ratio, val_ratio, and test_ratio should equal 1.0.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    with h5py.File(input_path, 'r') as f:
        num_samples = len(f['actions'])

        # Create random permutation
        rng = np.random.default_rng(random_seed)
        if shuffle:
            indices = rng.permutation(num_samples)
        else:
            indices = np.arange(num_samples)

        # Calculate split points
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Save splits
        for split_name, split_indices in [
            ('train', train_indices),
            ('val', val_indices),
            ('test', test_indices)
        ]:
            if len(split_indices) == 0:
                continue

            output_file = output_path / f'{split_name}.h5'

            with h5py.File(output_file, 'w') as out_f:
                # Copy data for this split
                out_f.create_dataset(
                    'spectrograms',
                    data=f['spectrograms'][split_indices]
                )
                out_f.create_dataset(
                    'actions',
                    data=f['actions'][split_indices]
                )

                # Copy optional datasets
                if 'positions' in f:
                    out_f.create_dataset(
                        'positions',
                        data=f['positions'][split_indices]
                    )

                # Copy maze (same for all splits)
                if 'maze' in f:
                    out_f.create_dataset('maze', data=f['maze'][:])

            print(f"Created {split_name} set: {len(split_indices)} samples -> {output_file}")
