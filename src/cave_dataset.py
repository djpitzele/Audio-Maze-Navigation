import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

# Defaults / shared constants
ACTION_MAP: Dict[str, int] = {
    'stop': 0,
    'up': 1,
    'down': 2,
    'left': 3,
    'right': 4,
    '': -1,  # Invalid/wall
}
ACTION_NAMES = ['STOP', 'UP', 'DOWN', 'LEFT', 'RIGHT']
MIC_OFFSETS = [
    (0, 1),   # Right
    (1, 1),   # Down-right
    (1, 0),   # Down
    (1, -1),  # Down-left
    (0, -1),  # Left
    (-1, -1), # Up-left
    (-1, 0),  # Up
    (-1, 1)   # Up-right
]


def _decode_actions(action_arr):
    if action_arr.dtype.kind == 'S':
        return np.vectorize(lambda x: x.decode('utf-8'))(action_arr)
    return action_arr.astype(str)


def compute_class_distribution(dataset) -> Dict[str, int]:
    counts = {name.lower(): 0 for name in ACTION_NAMES}
    for file_idx, y, x in dataset.valid_positions:
        a = dataset.file_infos[file_idx]['action_grid'][y, x]
        counts[a if isinstance(a, str) else a.decode('utf-8')] += 1
    return counts


def compute_class_weights(counts: Dict[str, int], method='sqrt') -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Args:
        counts: Dictionary of class name -> sample count
        method: 'inverse' (harsh), 'sqrt' (balanced), or 'log' (soft)

    Returns:
        Tensor of class weights
    """
    arr = np.array([counts.get(name.lower(), 1) for name in ACTION_NAMES], dtype=np.float32)
    total = arr.sum()

    if method == 'inverse':
        # Inverse frequency: weight_i = total / (n_classes * count_i)
        weights = total / (len(ACTION_NAMES) * arr)
    elif method == 'sqrt':
        # Square root smoothing (more balanced)
        # weight_i = sqrt(total / count_i)
        weights = np.sqrt(total / arr)
        # Normalize to sum to n_classes (optional, for stability)
        weights = weights / weights.sum() * len(ACTION_NAMES)
    elif method == 'log':
        # Log smoothing (softest)
        weights = np.log(total / arr + 1)
        weights = weights / weights.sum() * len(ACTION_NAMES)
    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.tensor(weights, dtype=torch.float32)


class MultiCaveDataset(Dataset):
    """
    PyTorch Dataset over multiple cave HDF5 files.

    Each sample is a valid agent position (3x3 footprint) with an 8-mic
    pressure time-series and the corresponding action label.
    """
    def __init__(self, file_paths: List[Path], agent_radius: int = 1,
                 mic_offsets: List[Tuple[int, int]] = None,
                 action_map: Dict[str, int] = None):
        self.file_paths = [Path(p) for p in file_paths]
        self.agent_radius = agent_radius
        self.mic_offsets = mic_offsets if mic_offsets else MIC_OFFSETS
        self.action_map = action_map if action_map else ACTION_MAP

        self.file_infos = []  # per-file cached arrays and metadata
        self.valid_positions = []  # list of (file_idx, y, x)
        self._file_handles = {}

        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as f:
                key = list(f.keys())[0]
                cave_grid = f[key]['cave_grid'][:]
                action_grid = _decode_actions(f[key]['action_grid'][:])
                pf_shape = f[key]['pressure_timeseries'].shape
                end_pos = tuple(f[key].attrs['end_position'])
                start_pos = tuple(f[key].attrs.get('start_position', (-1, -1)))

            Nx, Ny, _ = pf_shape
            valid = []
            for y in range(Nx):
                for x in range(Ny):
                    a = action_grid[y, x]
                    if a in self.action_map and a != '':
                        if self._is_valid_footprint(cave_grid, y, x):
                            valid.append((y, x))
                            self.valid_positions.append((file_idx, y, x))

            self.file_infos.append({
                'path': path,
                'key': key,
                'cave_grid': cave_grid,
                'action_grid': action_grid,
                'valid': valid,
                'end_pos': end_pos,
                'start_pos': start_pos,
                'shape': pf_shape,
            })

    def _is_valid_footprint(self, cave_grid, y, x):
        r = self.agent_radius
        if y - r < 0 or y + r >= cave_grid.shape[0] or x - r < 0 or x + r >= cave_grid.shape[1]:
            return False
        footprint = cave_grid[y - r:y + r + 1, x - r:x + r + 1]
        return np.all(footprint == 0)

    def _get_file(self, file_idx):
        if file_idx not in self._file_handles:
            path = self.file_infos[file_idx]['path']
            self._file_handles[file_idx] = h5py.File(path, 'r')
        return self._file_handles[file_idx]

    def __len__(self):
        return len(self.valid_positions)

    def __getitem__(self, idx):
        file_idx, y, x = self.valid_positions[idx]
        info = self.file_infos[file_idx]
        f = self._get_file(file_idx)
        g = f[info['key']]

        mic_data = []
        for dy, dx in self.mic_offsets:
            my, mx = y + dy, x + dx
            mic_data.append(g['pressure_timeseries'][my, mx, :])
        mic_data = np.array(mic_data, dtype=np.float32)

        # Per-sample normalization (critical for generalization)
        mic_mean = mic_data.mean()
        mic_std = mic_data.std()
        if mic_std > 1e-8:
            mic_data = (mic_data - mic_mean) / mic_std
        else:
            mic_data = mic_data - mic_mean

        action_str = info['action_grid'][y, x]
        action_label = self.action_map[action_str]

        return (
            torch.from_numpy(mic_data),
            torch.tensor(action_label, dtype=torch.long),
            torch.tensor(file_idx, dtype=torch.long),
            torch.tensor([y, x], dtype=torch.long),
        )

    def get_sample_with_position(self, idx):
        mic, action, file_idx, pos = self.__getitem__(idx)
        return mic, action, (int(pos[0]), int(pos[1])), int(file_idx)

    def close(self):
        for fh in self._file_handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._file_handles.clear()
