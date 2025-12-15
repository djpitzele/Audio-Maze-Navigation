"""
Convert HDF5 cave dataset to LMDB format for fast training.
FIXED: Handles Numpy int64 serialization error.
"""

import argparse
import h5py
import lmdb
import numpy as np
import pickle
import random
from pathlib import Path
from tqdm.auto import tqdm
import json
from collections import defaultdict

def decode_actions(action_arr):
    if action_arr.dtype.kind == 'S':
        return np.vectorize(lambda x: x.decode('utf-8'))(action_arr)
    return action_arr.astype(str)

def clean_numpy(obj):
    """Recursively convert numpy types to python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [clean_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    return obj

def convert_dataset_to_lmdb(input_dir, output_dir, agent_radius=1, map_size_gb=200):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob('cave_*.h5'))
    if len(h5_files) == 0:
        raise ValueError(f"No cave_*.h5 files found in {input_dir}")

    print(f"Found {len(h5_files)} HDF5 files")

    mic_offsets = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]

    action_map = {
        'stop': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4, '': -1,
    }
    balancing_keys = {'up', 'down', 'left', 'right'}

    def is_valid_footprint(cave_grid, y, x, radius):
        if y - radius < 0 or y + radius >= cave_grid.shape[0]: return False
        if x - radius < 0 or x + radius >= cave_grid.shape[1]: return False
        footprint = cave_grid[y - radius:y + radius + 1, x - radius:x + radius + 1]
        return np.all(footprint == 0)

    # PASS 1: SCANNING
    print("\n[Pass 1] Scanning dataset to calculate distribution...")
    samples_by_action = defaultdict(list)
    files_metadata = {} 

    for file_idx, path in enumerate(tqdm(h5_files, desc="Scanning files")):
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            cave_grid = f[key]['cave_grid'][:]
            action_grid = decode_actions(f[key]['action_grid'][:])
            pf_shape = f[key]['pressure_timeseries'].shape
            
            # Ensure attributes are converted to standard lists immediately
            end_pos = f[key].attrs['end_position']
            start_pos = f[key].attrs.get('start_position', (-1, -1))

            files_metadata[file_idx] = {
                'path': str(path), 'key': key,
                'end_pos': [int(x) for x in end_pos],   # Explicit cast
                'start_pos': [int(x) for x in start_pos], # Explicit cast
                'valid_count': 0
            }

        Nx, Ny, _ = pf_shape

        for y in range(Nx):
            for x in range(Ny):
                a = action_grid[y, x]
                if a in action_map and a != '':
                    if is_valid_footprint(cave_grid, y, x, agent_radius):
                        samples_by_action[a].append((file_idx, y, x))

    # BALANCING
    print("\n" + "-" * 40)
    print("Original Distribution:")
    counts = {k: len(v) for k, v in samples_by_action.items()}
    for action, count in counts.items():
        print(f"  {action:<10}: {count:,}")
    
    move_counts = [count for k, count in counts.items() if k in balancing_keys]
    target_count = min(move_counts) if move_counts else (min(counts.values()) if counts else 0)

    print("-" * 40)
    print(f"Target count per movement class: {target_count:,}")
    print("-" * 40)

    valid_samples_by_file = defaultdict(list)
    final_action_counts = defaultdict(int)

    for action, samples in samples_by_action.items():
        if action in balancing_keys:
            if len(samples) > target_count:
                kept_samples = random.sample(samples, target_count)
            else:
                kept_samples = samples
        else:
            kept_samples = samples

        final_action_counts[action] = len(kept_samples)
        for (file_idx, y, x) in kept_samples:
            valid_samples_by_file[file_idx].append((y, x, action))
            files_metadata[file_idx]['valid_count'] += 1

    total_samples = sum(final_action_counts.values())
    print(f"Balanced Total Samples: {total_samples:,}")

    # PASS 2: WRITING
    map_size_bytes = int(map_size_gb * 1024**3)
    print(f"\n[Pass 2] Writing to LMDB...")

    lmdb_path = output_dir / "data.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=map_size_bytes, writemap=True)

    sample_idx = 0
    with env.begin(write=True) as txn:
        sorted_file_indices = sorted(valid_samples_by_file.keys())
        for file_idx in tqdm(sorted_file_indices, desc="Writing balanced samples"):
            meta = files_metadata[file_idx]
            positions = valid_samples_by_file[file_idx]
            positions.sort(key=lambda p: (p[0], p[1]))

            with h5py.File(meta['path'], 'r') as f:
                g = f[meta['key']]
                pressure_data = g['pressure_timeseries']

                for y, x, action_str in positions:
                    mic_data = []
                    for dy, dx in mic_offsets:
                        mic_data.append(pressure_data[y+dy, x+dx, :])
                    
                    sample = {
                        'mic_data': np.array(mic_data, dtype=np.float32),
                        'action': action_map[action_str],
                        'file_idx': file_idx,
                        'position': (y, x),
                    }
                    txn.put(str(sample_idx).encode('ascii'), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                    sample_idx += 1

    # Metadata
    metadata = {
        'num_samples': int(sample_idx),
        'action_counts': {k: int(v) for k, v in final_action_counts.items()},
        'agent_radius': int(agent_radius),
        'mic_offsets': [[int(dy), int(dx)] for dy, dx in mic_offsets],
        'action_map': {k: int(v) for k, v in action_map.items()},
        'file_infos': list(files_metadata.values())
    }

    # SANITIZE METADATA BEFORE SAVING
    metadata = clean_numpy(metadata)

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    env.close()
    print("\n" + "=" * 70)
    print(f"CONVERSION COMPLETE. Saved to: {lmdb_path}")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='D:/audiomaze_dataset_100')
    parser.add_argument('--output_dir', type=str, default='D:/audiomaze_lmdb_100')
    parser.add_argument('--agent_radius', type=int, default=1)
    parser.add_argument('--map_size_gb', type=int, default=200)
    args = parser.parse_args()

    convert_dataset_to_lmdb(args.input_dir, args.output_dir, args.agent_radius, args.map_size_gb)