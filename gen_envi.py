import numpy as np
import matplotlib.pyplot as plt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst

# SETTINGS
SHOW_TIME_SNAPSHOTS = True  # Set False if you only want RMS/final results
Nx, Ny = 100, 100  # Grid size
cell_size = 10
portion_blocks = 0.2  # Portion of grid to fill with maze segments
dx = dy = 1e-2  # 1 cm resolution
sim_len = 4e-3  # Simulation time

# CONSTANTS
c_air, rho_air = 343.0, 1.2
c_obj = 220.0
alpha_power = 1.5
f0 = 2e3  # Hz
target_db_per_cm = 1.0  # Desired loss at f0 inside objects

def add_rect(obj_mask, y0, x0, h, w):
    obj_mask[y0:y0 + h, x0:x0 + w] = True
    return obj_mask

def create_source(kgrid, sy, sx):
    src = kSource()
    src_mask = np.zeros(kgrid.N, dtype=bool)
    src_mask[sy, sx] = True
    src.p_mask = src_mask
    src.p = tone_burst(1 / kgrid.dt, f0, 6)
    return src

def create_sensor(kgrid, rec_list):
    sensor_mask = np.ones(kgrid.N, dtype=bool)
    return kSensor(mask=sensor_mask, record=rec_list)

def gen_random_maze():
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # Background (air)
    rho = np.full((Nx, Ny), rho_air)

    # Random maze segments
    rng = np.random.default_rng(2)
    obj_mask = np.zeros((Nx, Ny), dtype=bool)

    for _ in range(int(portion_blocks * (Nx * Ny) / (cell_size ** 2))):
        h, w = cell_size, cell_size
        # make the rectangles always appear on the grid
        
        y0, x0 = cell_size * rng.integers(0, (Nx // cell_size)), cell_size * rng.integers(0, (Ny // cell_size))
        add_rect(obj_mask, y0, x0, h, w)

    # Material maps
    c_with = np.full((Nx, Ny), c_air)
    c_with[obj_mask] = c_obj

    alpha_with = np.zeros((Nx, Ny))
    alpha_with[obj_mask] = target_db_per_cm / ((f0 / 1e6) ** alpha_power)

    # Media definitions
    medium_objs = kWaveMedium(sound_speed=c_with, density=rho, alpha_coeff=alpha_with, alpha_power=alpha_power)
    medium_empty = kWaveMedium(sound_speed=np.full((Nx, Ny), c_air), density=rho,
                            alpha_coeff=np.zeros((Nx, Ny)), alpha_power=alpha_power)

    # Time array
    kgrid.makeTime(c=c_air, cfl=0.3, t_end=sim_len)

    # Source placement
    sy = int(Nx * 0.95) # make more precise with grid
    sx = int(Ny * 0.95)

    rec_list = ['p_final', 'p_rms'] + (['p'] if SHOW_TIME_SNAPSHOTS else [])

    # Simulation options
    sim_opts = SimulationOptions(save_to_disk=True, pml_inside=False)
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False, show_sim_log=False)

    # Quick sanity check on shapes
    preview_source = create_source(kgrid, sy, sx)
    preview_sensor = create_sensor(kgrid, rec_list)
    print(f"Source mask shape: {preview_source.p_mask.shape}, expected: {kgrid.N}")
    print(f"Sensor mask shape: {preview_sensor.mask.shape}, expected: {kgrid.N}")

    # Run simulations
    res_objs = kspaceFirstOrder2D(kgrid, create_source(kgrid, sy, sx), create_sensor(kgrid, rec_list), medium_objs, sim_opts, exec_opts)
    p_final_obj = np.reshape(res_objs['p_final'], (Nx, Ny), order='F')
    p_rms_obj = np.reshape(res_objs['p_rms'], (Nx, Ny), order='F')
    p_time_obj = res_objs['p'] if SHOW_TIME_SNAPSHOTS else None

    res_empty = kspaceFirstOrder2D(kgrid, create_source(kgrid, sy, sx), create_sensor(kgrid, rec_list), medium_empty, sim_opts, exec_opts)
    p_final_empty = np.reshape(res_empty['p_final'], (Nx, Ny), order='F')
    p_rms_empty = np.reshape(res_empty['p_rms'], (Nx, Ny), order='F')
    p_time_empty = res_empty['p'] if SHOW_TIME_SNAPSHOTS else None

    delta_rms = p_rms_obj - p_rms_empty
    delta_final = p_final_obj - p_final_empty

    # Visualize medium
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(c_with.T, origin='lower')
    plt.title('Sound speed map (objects)')
    plt.axis('image')
    plt.colorbar(label='m/s')
    plt.tight_layout()
    plt.show()

    # Visualize RMS Fields
    obj_contour = obj_mask
    plt.figure(figsize=(15, 4.6))

    plt.subplot(1, 3, 1)
    plt.imshow(p_rms_empty, origin='lower')
    plt.title('RMS - empty room')
    plt.axis('image')
    plt.colorbar()
    plt.scatter([sx], [sy], s=60, c='r', marker='*', edgecolors='k', linewidths=0.5)

    plt.subplot(1, 3, 2)
    plt.imshow(p_rms_obj, origin='lower')
    plt.title('RMS - with objects')
    plt.axis('image')
    plt.colorbar()
    plt.contour(obj_contour, levels=[0.5], colors='k', linewidths=1)
    plt.scatter([sx], [sy], s=60, c='r', marker='*', edgecolors='k', linewidths=0.5)

    plt.subplot(1, 3, 3)
    plt.imshow(delta_rms, origin='lower', cmap='RdBu')
    plt.title('RMS difference (objects - empty)')
    plt.axis('image')
    plt.colorbar()
    plt.contour(obj_contour, levels=[0.5], colors='k', linewidths=1)
    plt.scatter([sx], [sy], s=60, c='r', marker='*', edgecolors='k', linewidths=0.5)

    plt.tight_layout()
    plt.show()

    # Visualize Final Pressure Fields
    plt.figure(figsize=(15, 4.6))

    plt.subplot(1, 3, 1)
    plt.imshow(p_final_empty, origin='lower', cmap='RdBu', vmin=-0.08, vmax=0.08)
    plt.title('Final - empty')
    plt.axis('image')
    plt.colorbar()
    plt.scatter([sx], [sy], s=60, c='r', marker='*', edgecolors='k', linewidths=0.5)

    plt.subplot(1, 3, 2)
    plt.imshow(p_final_obj, origin='lower', cmap='RdBu', vmin=-0.08, vmax=0.08)
    plt.title('Final - with objects')
    plt.axis('image')
    plt.colorbar()
    plt.contour(obj_contour, levels=[0.5], colors='k', linewidths=1)
    plt.scatter([sx], [sy], s=60, c='r', marker='*', edgecolors='k', linewidths=0.5)

    plt.subplot(1, 3, 3)
    plt.imshow(delta_final, origin='lower', cmap='RdBu', vmin=-0.05, vmax=0.05)
    plt.title('Final difference')
    plt.axis('image')
    plt.colorbar()
    plt.contour(obj_contour, levels=[0.5], colors='k', linewidths=1)
    plt.scatter([sx], [sy], s=60, c='r', marker='*', edgecolors='k', linewidths=0.5)

    plt.tight_layout()
    plt.show()

"""## Optional: Time Snapshots"""

# if SHOW_TIME_SNAPSHOTS and p_time_obj is not None:
#     print(f"p_time_obj shape: {p_time_obj.shape}")

#     Nt = p_time_obj.shape[0]
#     snaps = np.linspace(0, Nt - 1, 3, dtype=int)

#     plt.figure(figsize=(12, 3.8))
#     for i, t in enumerate(snaps):
#         plt.subplot(1, 3, i + 1)
#         frame = np.reshape(p_time_obj[t, :], (Nx, Ny), order='F')
#         plt.imshow(frame, origin='lower', cmap='RdBu', vmin=-0.08, vmax=0.08)
#         plt.contour(obj_contour, levels=[0.5], colors='k', linewidths=1)
#         plt.scatter([sx], [sy], s=50, c='r', marker='*', edgecolors='k', linewidths=0.4)
#         plt.title(f"t = {t * kgrid.dt * 1e3:.2f} ms")
#         plt.axis('image')
#     plt.tight_layout()
#     plt.show()
if __name__ == "__main__":
    gen_random_maze()