"""Generate large batch of acoustic cave datasets - SEQUENTIAL to avoid k-Wave temp file collisions."""
import sys
sys.path.append('data')

import os
import numpy as np
import h5py
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Thread limiting per process (leave headroom for system)
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'

from audio_cave_sim import AudioCaveSim

# Dataset parameters
DATASET_DIR = r"D:\audiomaze_dataset"
NUM_SAMPLES = 100
GRID_SIZE = 60
RESOLUTION = 0.01  # meters

def generate_single_cave(sample_idx):
    """Generate a single cave simulation and save to HDF5.

    Args:
        sample_idx: Index of the sample (0-999)

    Returns:
        Tuple of (sample_idx, success_bool, message)
    """
    start_time = time.time()
    try:
        # IMPORTANT: Set unique temp directory for this process to avoid k-Wave file collisions
        # k-Wave uses timestamp-based temp filenames, so we need process-specific temp dirs
        unique_temp_dir = os.path.join(tempfile.gettempdir(), f'kwave_{os.getpid()}_{sample_idx}')
        os.makedirs(unique_temp_dir, exist_ok=True)
        old_temp = os.environ.get('TEMP')
        old_tmp = os.environ.get('TMP')
        os.environ['TEMP'] = unique_temp_dir
        os.environ['TMP'] = unique_temp_dir

        try:
            # Create cave simulation (runs automatically in __init__)
            cave = AudioCaveSim(Nx=GRID_SIZE, Ny=GRID_SIZE, res=RESOLUTION)

            # Reshape pressure field from (Nt, Nx*Ny) to (Nx, Ny, Nt)
            pressure_timeseries = cave.sensor_data['p'].T  # (Nt, Nx*Ny) -> (Nx*Ny, Nt)
            pressure_field = np.reshape(pressure_timeseries.T, (GRID_SIZE, GRID_SIZE, cave.kgrid.Nt), order='C')

            # Prepare output path
            output_path = os.path.join(DATASET_DIR, f'cave_{sample_idx:04d}.h5')

            # Save to HDF5
            with h5py.File(output_path, 'w') as f:
                cave_group = f.create_group(f'cave_{sample_idx:04d}')

                # Save cave structure
                cave_group.create_dataset('cave_grid', data=cave.am.grid.astype(np.int32), compression='gzip')
                cave_group.create_dataset('action_grid', data=np.array(cave.am.action_grid, dtype='S10'), compression='gzip')

                # Save pressure time-series (Nx, Ny, Nt)
                cave_group.create_dataset(
                    'pressure_timeseries',
                    data=pressure_field.astype(np.float32),
                    compression='gzip',
                    compression_opts=4
                )

                # Save metadata
                cave_group.attrs['grid_size'] = [GRID_SIZE, GRID_SIZE]
                cave_group.attrs['resolution_m'] = RESOLUTION
                cave_group.attrs['dt'] = cave.kgrid.dt
                cave_group.attrs['frequency_hz'] = cave.f0
                cave_group.attrs['num_timesteps'] = cave.kgrid.Nt
                cave_group.attrs['sim_duration_s'] = cave.sim_duration
                cave_group.attrs['sound_speed_air'] = cave.c_air
                cave_group.attrs['sound_speed_wall'] = cave.c_wall
                cave_group.attrs['start_position'] = cave.am.start
                cave_group.attrs['end_position'] = cave.am.end
                cave_group.attrs['generation_timestamp'] = datetime.now().isoformat()

            elapsed = time.time() - start_time
            return (sample_idx, True, f"Successfully generated cave {sample_idx:04d} in {elapsed:.1f}s")

        finally:
            # Restore original temp directories
            if old_temp:
                os.environ['TEMP'] = old_temp
            if old_tmp:
                os.environ['TMP'] = old_tmp

            # Clean up temp directory
            try:
                import shutil
                shutil.rmtree(unique_temp_dir, ignore_errors=True)
            except:
                pass

    except Exception as e:
        import traceback
        return (sample_idx, False, f"Failed cave {sample_idx:04d}: {str(e)}\n{traceback.format_exc()}")

def main():
    """Main function to generate dataset SEQUENTIALLY (avoids k-Wave temp file collisions)."""
    print("=" * 80)
    print("ACOUSTIC CAVE DATASET GENERATOR (SEQUENTIAL)")
    print("=" * 80)
    print(f"Output directory: {DATASET_DIR}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Resolution: {RESOLUTION} m")
    print(f"Mode: SEQUENTIAL (to avoid k-Wave temp file conflicts)")
    print("=" * 80)

    # Create output directory if it doesn't exist
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)

    # Check how many samples already exist
    existing_files = list(Path(DATASET_DIR).glob('cave_*.h5'))
    if existing_files:
        print(f"\nFound {len(existing_files)} existing cave files in output directory")
        print("These will be overwritten if they have the same indices.")

    print(f"\nStarting sequential generation...")
    print("This will take a while. Progress will be shown below.\n")

    start_time = datetime.now()

    # Generate sequentially to avoid k-Wave temp file collisions
    completed = 0
    failed = 0

    for idx in range(NUM_SAMPLES):
        print(f"\n{'='*60}")
        print(f"Generating cave {idx+1}/{NUM_SAMPLES} (index {idx:04d})...")
        print(f"{'='*60}")

        sample_idx, success, message = generate_single_cave(idx)

        if success:
            completed += 1
            print(f"✓ Cave {sample_idx:04d} completed successfully")
        else:
            failed += 1
            print(f"✗ ERROR: {message}")

        # Print progress summary
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (completed + failed) / elapsed if elapsed > 0 else 0
        remaining = (NUM_SAMPLES - completed - failed) / rate if rate > 0 else 0

        print(f"\n--- Overall Progress ---")
        print(f"Completed: {completed + failed}/{NUM_SAMPLES} ({100*(completed + failed)/NUM_SAMPLES:.1f}%)")
        print(f"Success: {completed} | Failed: {failed}")
        print(f"Rate: {rate:.3f} samples/s | ETA: {remaining/60:.1f} min")
        print(f"Elapsed: {elapsed/60:.1f} min")

    # Final summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total samples: {NUM_SAMPLES}")
    print(f"Successful: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed/60:.2f} minutes ({elapsed:.0f}s)")
    if completed > 0:
        print(f"Average time per sample: {elapsed/completed:.1f}s")
    print(f"Output directory: {DATASET_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
