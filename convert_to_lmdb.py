"""
Convert HDF5 cave dataset to LMDB format for fast training.

LMDB (Lightning Memory-Mapped Database) provides:
- Fast initialization (<1 second vs minutes for HDF5)
- Memory-mapped access (OS handles caching automatically)
- Single file format (instead of 100s of HDF5 files)
- Multi-worker safe (each worker can read independently)
- Scales to large datasets (1000+ mazes, 150GB+)

Usage:
    python convert_to_lmdb.py --input_dir D:/audiomaze_dataset_100 --output_dir D:/audiomaze_lmdb_100

    # Or with default paths:
    python convert_to_lmdb.py
"""

import argparse
import h5py
import lmdb
import numpy as np
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import json


def decode_actions(action_arr):
    """Decode action array from bytes to strings."""
    if action_arr.dtype.kind == 'S':
        return np.vectorize(lambda x: x.decode('utf-8'))(action_arr)
    return action_arr.astype(str)


def convert_dataset_to_lmdb(input_dir, output_dir, agent_radius=1, map_size_gb=200):
    """
    Convert HDF5 dataset to LMDB format.

    Args:
        input_dir: Directory containing cave_*.h5 files
        output_dir: Directory to save LMDB database
        agent_radius: Agent footprint radius (default: 1 for 3x3 footprint)
        map_size_gb: Maximum LMDB size in GB (default: 200GB)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all HDF5 files
    h5_files = sorted(input_dir.glob('cave_*.h5'))
    if len(h5_files) == 0:
        raise ValueError(f"No cave_*.h5 files found in {input_dir}")

    print(f"Found {len(h5_files)} HDF5 files")
    print(f"Output directory: {output_dir}")

    # Microphone offsets (circular 8-mic array)
    mic_offsets = [
        (0, 1),   # Right
        (1, 1),   # Down-right
        (1, 0),   # Down
        (1, -1),  # Down-left
        (0, -1),  # Left
        (-1, -1), # Up-left
        (-1, 0),  # Up
        (-1, 1)   # Up-right
    ]

    # Action mapping
    action_map = {
        'stop': 0,
        'up': 1,
        'down': 2,
        'left': 3,
        'right': 4,
        '': -1,
    }

    def is_valid_footprint(cave_grid, y, x, radius):
        """Check if agent footprint is fully in air."""
        if y - radius < 0 or y + radius >= cave_grid.shape[0]:
            return False
        if x - radius < 0 or x + radius >= cave_grid.shape[1]:
            return False
        footprint = cave_grid[y - radius:y + radius + 1, x - radius:x + radius + 1]
        return np.all(footprint == 0)

    # First pass: collect metadata and count valid positions
    print("\nScanning dataset for valid positions...")
    file_infos = []
    total_valid = 0

    for file_idx, path in enumerate(tqdm(h5_files, desc="Scanning files")):
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            cave_grid = f[key]['cave_grid'][:]
            action_grid = decode_actions(f[key]['action_grid'][:])
            pf_shape = f[key]['pressure_timeseries'].shape
            end_pos = tuple(f[key].attrs['end_position'])
            start_pos = tuple(f[key].attrs.get('start_position', (-1, -1)))

        Nx, Ny, _ = pf_shape
        valid_positions = []

        for y in range(Nx):
            for x in range(Ny):
                a = action_grid[y, x]
                if a in action_map and a != '':
                    if is_valid_footprint(cave_grid, y, x, agent_radius):
                        valid_positions.append((y, x))

        file_infos.append({
            'path': str(path),
            'key': key,
            'valid_positions': valid_positions,
            'end_pos': end_pos,
            'start_pos': start_pos,
        })
        total_valid += len(valid_positions)

    print(f"\nTotal valid positions: {total_valid:,}")

    # Create LMDB database
    map_size_bytes = int(map_size_gb * 1024**3)
    print(f"Creating LMDB database (max size: {map_size_gb}GB)...")

    lmdb_path = output_dir / "data.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_bytes,
        readonly=False,
        metasync=False,
        sync=False,
        writemap=True,
    )

    # Second pass: write samples to LMDB
    print("\nWriting samples to LMDB...")
    sample_idx = 0
    action_counts = {'stop': 0, 'up': 0, 'down': 0, 'left': 0, 'right': 0}

    with env.begin(write=True) as txn:
        for file_idx, info in enumerate(tqdm(file_infos, desc="Converting files")):
            with h5py.File(info['path'], 'r') as f:
                g = f[info['key']]
                action_grid = decode_actions(g['action_grid'][:])

                for y, x in info['valid_positions']:
                    # Extract 8-mic data
                    mic_data = []
                    for dy, dx in mic_offsets:
                        my, mx = y + dy, x + dx
                        mic_data.append(g['pressure_timeseries'][my, mx, :])
                    mic_data = np.array(mic_data, dtype=np.float32)

                    # Get action label
                    action_str = action_grid[y, x]
                    action_label = action_map[action_str]
                    action_counts[action_str] += 1

                    # Store sample
                    sample = {
                        'mic_data': mic_data,
                        'action': action_label,
                        'file_idx': file_idx,
                        'position': (y, x),
                    }

                    txn.put(
                        str(sample_idx).encode('ascii'),
                        pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                    sample_idx += 1

    # Save metadata
    metadata = {
        'num_samples': sample_idx,
        'num_files': len(file_infos),
        'action_counts': action_counts,
        'agent_radius': agent_radius,
        'mic_offsets': mic_offsets,
        'action_map': action_map,
        'file_infos': file_infos,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    env.close()

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"LMDB database: {lmdb_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Total samples: {sample_idx:,}")
    print(f"Action distribution: {action_counts}")
    print(f"Database size: {lmdb_path.stat().st_size / (1024**3):.2f} GB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 cave dataset to LMDB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default paths
  python convert_to_lmdb.py

  # Convert with custom paths
  python convert_to_lmdb.py --input_dir D:/audiomaze_dataset_100 --output_dir D:/audiomaze_lmdb_100

  # For larger datasets (1000 mazes)
  python convert_to_lmdb.py --input_dir D:/audiomaze_dataset_1000 --output_dir D:/audiomaze_lmdb_1000 --map_size_gb 300
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        default='D:/audiomaze_dataset_100',
        help='Directory containing cave_*.h5 files (default: D:/audiomaze_dataset_100)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='D:/audiomaze_lmdb_100',
        help='Directory to save LMDB database (default: D:/audiomaze_lmdb_100)'
    )

    parser.add_argument(
        '--agent_radius',
        type=int,
        default=1,
        help='Agent footprint radius (default: 1 for 3x3 footprint)'
    )

    parser.add_argument(
        '--map_size_gb',
        type=int,
        default=200,
        help='Maximum LMDB size in GB (default: 200). Use 300+ for 1000 mazes.'
    )

    args = parser.parse_args()

    convert_dataset_to_lmdb(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        agent_radius=args.agent_radius,
        map_size_gb=args.map_size_gb
    )


if __name__ == '__main__':
    main()
