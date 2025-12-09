import numpy as np
import matplotlib.pyplot as plt
from audio_maze import AudioMaze

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst

class AudioMazeSim:
    def __init__(self, Nx, Ny, res):
        self.Nx = Nx # 4n + 1 for some n
        self.Ny = Ny # 4n + 1 for some n
        self.dx = res  # 1cm resolution
        self.dy = res
        
        self.am = AudioMaze(Nx // 2, Ny // 2) # for some reason mazelib doubles dimensions (2n + 1)
        self.define_constants()
        print("Initializing simulation")
        self.init_sim()
        print("Initializing source and sensor")
        self.init_source_sensor()
        print("Running simulation")
        self.run_sim()
    
    def define_constants(self):
        # physical constants
        self.c_air = 343.0      # Sound speed in air (m/s)
        self.rho_air = 1.2      # Air density (kg/m³)
        self.c_wall = 220.0     # Sound speed in walls (slower = more reflection)
        self.alpha_power = 1.5  # Frequency power law for absorption
        self.wall_absorption_db_per_cm = 1.0  # Wall absorption (dB per cm at f0)
        
        # source settings
        self.f0 = 15000.0        # 15 kHz frequency (good for 17x17 mazes)
        self.num_cycles = 6     # Tone burst cycles
        self.sim_duration = 4e-3  # 4 ms simulation
        
    def init_sim(self):
        self.kgrid = kWaveGrid([self.Nx, self.Ny], [self.dx, self.dy])
        self.kgrid.makeTime(c=self.c_air, cfl=0.3, t_end=self.sim_duration)
        
        # create medium properties
        self.sound_speed = np.full((self.Nx, self.Ny), self.c_air, dtype=np.float32)
        self.sound_speed[self.am.maze.grid == 1] = self.c_wall
        density = np.full((self.Nx, self.Ny), self.rho_air, dtype=np.float32)

        # Absorption (walls absorb sound)
        alpha_coeff = np.zeros((self.Nx, self.Ny), dtype=np.float32)
        alpha_coeff[self.am.maze.grid == 1] = self.wall_absorption_db_per_cm / ((self.f0 / 1e6) ** self.alpha_power)

        self.medium = kWaveMedium(
            sound_speed=self.sound_speed,
            density=density,
            alpha_coeff=alpha_coeff,
            alpha_power=self.alpha_power
        )
    
    def init_source_sensor(self):
        self.source_x = self.Nx - 2
        self.source_y = self.Ny - 2

        self.source = kSource()
        source_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        source_mask[self.source_y, self.source_x] = True
        self.source.p_mask = source_mask

        # Generate tone burst signal
        self.source.p = tone_burst(1 / self.kgrid.dt, self.f0, self.num_cycles)

        # Create sensor - record FULL pressure field TIME-SERIES
        # CRITICAL: Recording 'p' for time-series (needed for spectrograms)
        # NOT 'p_final'/'p_rms' which are aggregated values
        self.sensor = kSensor(
            mask=np.ones((self.Nx, self.Ny), dtype=bool),  # Record everywhere!
            record=['p']  # Pressure time-series at each point
        )

    def run_sim(self):
        sim_options = SimulationOptions(
            save_to_disk=True,
            pml_inside=False,
            pml_size=10
        )

        exec_options = SimulationExecutionOptions(
            is_gpu_simulation=False,
            show_sim_log=True  # Show progress
        )

        self.sensor_data = kspaceFirstOrder2D(
            kgrid=self.kgrid,
            source=self.source,
            sensor=self.sensor,
            medium=self.medium,
            simulation_options=sim_options,
            execution_options=exec_options
        )

    def visualize_sound_speed(self):
        # Visualize sound speed
        plt.figure(figsize=(7, 6))
        plt.imshow(self.sound_speed.T, origin='lower', cmap='viridis')
        plt.title('Sound Speed Map')
        plt.xlabel('X (grid points)')
        plt.ylabel('Y (grid points)')
        plt.colorbar(label='Sound Speed (m/s)')
        plt.axis('image')
        plt.tight_layout()
        plt.show()
        
    def visualize_wave(self):
        # Reshape pressure time-series to 3D: (Nx, Ny, Nt)
        pressure_timeseries = self.sensor_data['p'].T
        pressure_field = np.reshape(pressure_timeseries.T, (self.Nx, self.Ny, self.kgrid.Nt), order='F')

        # Compute p_final and p_rms from time-series for visualization
        p_final = pressure_field[:, :, -1]
        p_rms = np.sqrt(np.mean(pressure_field**2, axis=2))

        print(f"Pressure field shape: {pressure_field.shape}")
        print(f"Final pressure range: [{p_final.min():.3f}, {p_final.max():.3f}]")
        print(f"RMS pressure range: [{p_rms.min():.6f}, {p_rms.max():.6f}]")

        # # Check if sound reached borders
        # border_rms = np.concatenate([
        #     p_rms[0, :],   # Top
        #     p_rms[-1, :],  # Bottom
        #     p_rms[:, 0],   # Left
        #     p_rms[:, -1]   # Right
        # ])

        # print(f"\nBorder RMS mean: {border_rms.mean():.6f}")
        # print(f"Border RMS max: {border_rms.max():.6f}")

        # if border_rms.max() > p_rms.max() * 0.01:
        #     print("✓ Sound reached borders!")
        # else:
        #     print("⚠ Sound may not have reached borders (increase sim_duration)")

        # Plot Final Pressure Field (snapshot at end of simulation)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Final pressure
        im1 = axes[0].imshow(p_final.T, origin='lower', cmap='RdBu_r', 
                            vmin=-p_final.max()*0.8, vmax=p_final.max()*0.8)
        axes[0].contour(self.am.maze.grid.T, levels=[0.5], colors='black', linewidths=1.5)
        axes[0].scatter([self.source_x], [self.source_y], s=200, c='red', marker='*', 
                    edgecolors='black', linewidths=1, label='Source', zorder=10)
        axes[0].set_title('Final Pressure Field (Wave Pattern)', fontsize=14)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].legend(loc='upper left')
        axes[0].axis('image')
        plt.colorbar(im1, ax=axes[0], label='Pressure')

        # RMS pressure (shows energy distribution)
        im2 = axes[1].imshow(p_rms.T, origin='lower', cmap='hot')
        axes[1].contour(self.am.maze.grid.T, levels=[0.5], colors='cyan', linewidths=1.5)
        axes[1].scatter([self.source_x], [self.source_y], s=200, c='red', marker='*',
                    edgecolors='black', linewidths=1, label='Source', zorder=10)
        axes[1].set_title('RMS Pressure (Energy Distribution)', fontsize=14)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend(loc='upper left')
        axes[1].axis('image')
        plt.colorbar(im2, ax=axes[1], label='RMS Pressure')

        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    sim = AudioMazeSim(Nx=17, Ny=17, res=0.01)
    sim.visualize_sound_speed()
    sim.visualize_wave()