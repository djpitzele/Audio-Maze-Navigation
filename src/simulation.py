"""
Acoustic Simulation Module using k-Wave

This module provides a rigorous acoustic simulation framework using the k-Wave
library for physically accurate wave propagation in 2D maze environments.

Key Physics:
- Air: c = 343 m/s, ρ = 1.2 kg/m³
- Walls: c = 2500 m/s, ρ = 2000 kg/m³ (concrete-like material)
- Acoustic impedance mismatch creates realistic reflections
"""

import numpy as np
import warnings
from typing import Tuple, Optional

# Import k-Wave modules
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
    from kwave.options.simulation_options import SimulationOptions
except ImportError as e:
    raise ImportError(
        "k-wave-python is required for acoustic simulations. "
        "Install it with: pip install k-wave-python"
    ) from e


class AcousticSimulator:
    """
    High-fidelity acoustic simulator using k-Wave for 2D maze navigation.

    This class handles the full physics-based simulation of sound propagation
    through a maze environment, including:
    - Wave equation solving via k-space pseudospectral methods
    - Acoustic impedance boundaries (air/wall interfaces)
    - Multi-microphone sensor arrays
    - Broadband pulse excitation

    Parameters
    ----------
    grid_spacing : float, optional
        Physical spacing between grid points in meters (default: 0.01 m = 1 cm)
    time_step : float, optional
        Simulation time step in seconds. If None, uses CFL-limited automatic value.
    simulation_duration : float, optional
        Total simulation time in seconds (default: 0.02 s = 20 ms)
    source_frequency : float, optional
        Center frequency of the source pulse in Hz (default: 5000 Hz)
    num_cycles : int, optional
        Number of cycles in the tone burst (default: 3)
    pml_size : int, optional
        Size of the Perfectly Matched Layer for boundary absorption (default: 10)
    microphone_array_radius : int, optional
        Radius of circular microphone array in grid points (default: 2)
    num_microphones : int, optional
        Number of microphones in the circular array (default: 8)

    Attributes
    ----------
    AIR_SPEED : float
        Speed of sound in air (343 m/s)
    AIR_DENSITY : float
        Density of air (1.2 kg/m³)
    WALL_SPEED : float
        Speed of sound in wall material (2500 m/s)
    WALL_DENSITY : float
        Density of wall material (2000 kg/m³)
    """

    # Physical constants for materials
    AIR_SPEED = 343.0  # m/s
    AIR_DENSITY = 1.2  # kg/m³
    WALL_SPEED = 2500.0  # m/s (concrete-like)
    WALL_DENSITY = 2000.0  # kg/m³

    def __init__(
        self,
        grid_spacing: float = 0.01,  # 1 cm
        time_step: Optional[float] = None,
        simulation_duration: float = 0.02,  # 20 ms
        source_frequency: float = 5000.0,  # 5 kHz
        num_cycles: int = 3,
        pml_size: int = 10,
        microphone_array_radius: int = 2,
        num_microphones: int = 8,
    ):
        self.grid_spacing = grid_spacing
        self.time_step = time_step
        self.simulation_duration = simulation_duration
        self.source_frequency = source_frequency
        self.num_cycles = num_cycles
        self.pml_size = pml_size
        self.microphone_array_radius = microphone_array_radius
        self.num_microphones = num_microphones

    def _create_medium(
        self,
        maze: np.ndarray
    ) -> kWaveMedium:
        """
        Create a k-Wave medium with spatially varying acoustic properties.

        The maze array (0=air, 1=wall) is mapped to continuous fields of
        sound speed and density, creating acoustic impedance contrasts
        that produce realistic reflections.

        Parameters
        ----------
        maze : np.ndarray
            2D binary array where 0 = air, 1 = wall

        Returns
        -------
        kWaveMedium
            Configured k-Wave medium object
        """
        # Initialize arrays for sound speed and density
        sound_speed = np.zeros_like(maze, dtype=np.float32)
        density = np.zeros_like(maze, dtype=np.float32)

        # Map maze values to physical properties
        # Air regions (0)
        air_mask = (maze == 0)
        sound_speed[air_mask] = self.AIR_SPEED
        density[air_mask] = self.AIR_DENSITY

        # Wall regions (1)
        wall_mask = (maze == 1)
        sound_speed[wall_mask] = self.WALL_SPEED
        density[wall_mask] = self.WALL_DENSITY

        # Create and configure the medium
        medium = kWaveMedium(
            sound_speed=sound_speed,
            density=density,
        )

        return medium

    def _create_source(
        self,
        grid: kWaveGrid,
        source_pos: Tuple[int, int]
    ) -> kSource:
        """
        Create a point source with a tone burst signal.

        The source emits a Gaussian-windowed sinusoidal pulse, providing
        broadband frequency content for rich acoustic signatures.

        Parameters
        ----------
        grid : kWaveGrid
            k-Wave grid object
        source_pos : Tuple[int, int]
            (row, col) position of the source in grid coordinates

        Returns
        -------
        kSource
            Configured k-Wave source object
        """
        # Create source mask (single point)
        source_mask = np.zeros((grid.Nx, grid.Ny), dtype=bool)
        source_mask[source_pos[0], source_pos[1]] = True

        # Generate tone burst signal
        # Time vector
        t_array = np.arange(0, grid.Nt) * grid.dt

        # Gaussian-windowed sinusoid (tone burst)
        signal = np.sin(2 * np.pi * self.source_frequency * t_array)

        # Gaussian window
        tone_burst_duration = self.num_cycles / self.source_frequency
        tone_burst_std = tone_burst_duration / 4  # 4-sigma window
        gaussian_window = np.exp(
            -((t_array - tone_burst_duration / 2) ** 2) / (2 * tone_burst_std ** 2)
        )

        signal *= gaussian_window

        # Normalize
        signal = signal / np.max(np.abs(signal))

        # Create source
        source = kSource()
        source.p_mask = source_mask
        source.p = signal

        return source

    def _create_sensor_array(
        self,
        grid: kWaveGrid,
        agent_pos: Tuple[int, int]
    ) -> kSensor:
        """
        Create a circular microphone array centered at the agent position.

        The array provides directional acoustic information, enabling the
        agent to distinguish reflections from different directions.

        Parameters
        ----------
        grid : kWaveGrid
            k-Wave grid object
        agent_pos : Tuple[int, int]
            (row, col) position of the agent in grid coordinates

        Returns
        -------
        kSensor
            Configured k-Wave sensor object
        """
        # Create sensor mask
        sensor_mask = np.zeros((grid.Nx, grid.Ny), dtype=bool)

        # Place microphones in a circle around agent position
        angles = np.linspace(0, 2 * np.pi, self.num_microphones, endpoint=False)

        for angle in angles:
            # Calculate microphone position
            mic_row = int(agent_pos[0] + self.microphone_array_radius * np.cos(angle))
            mic_col = int(agent_pos[1] + self.microphone_array_radius * np.sin(angle))

            # Ensure within bounds
            mic_row = np.clip(mic_row, 0, grid.Nx - 1)
            mic_col = np.clip(mic_col, 0, grid.Ny - 1)

            sensor_mask[mic_row, mic_col] = True

        # Create sensor
        sensor = kSensor()
        sensor.mask = sensor_mask

        return sensor

    def run_simulation(
        self,
        maze: np.ndarray,
        agent_pos: Tuple[int, int],
        source_pos: Tuple[int, int],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Run a complete k-Wave acoustic simulation.

        This method orchestrates the full simulation pipeline:
        1. Create computational grid
        2. Define medium properties from maze
        3. Configure source and sensors
        4. Execute k-space pseudospectral solver
        5. Return time-series sensor data

        Parameters
        ----------
        maze : np.ndarray
            2D binary array (0=air, 1=wall)
        agent_pos : Tuple[int, int]
            Agent position (row, col) in grid coordinates
        source_pos : Tuple[int, int]
            Sound source position (row, col) in grid coordinates
        verbose : bool, optional
            If True, print simulation progress (default: False)

        Returns
        -------
        np.ndarray
            Sensor time-series data with shape (num_microphones, num_time_steps)
            Each row contains the pressure signal recorded at one microphone.

        Notes
        -----
        This simulation solves the acoustic wave equation using k-Wave's
        k-space pseudospectral method, which provides spectral accuracy
        and efficient computation compared to finite difference methods.

        The simulation is intentionally slow because it solves coupled
        PDEs at each time step. For rapid prototyping, use small mazes
        (e.g., 20x20) and short simulation times.
        """
        # Create k-Wave grid
        Nx, Ny = maze.shape

        grid = kWaveGrid([Nx, Ny], [self.grid_spacing, self.grid_spacing])

        # Calculate number of time steps
        if self.time_step is None:
            # Use CFL condition: dt <= dx / (c_max * sqrt(2))
            c_max = self.WALL_SPEED  # Maximum sound speed
            dt = 0.3 * self.grid_spacing / (c_max * np.sqrt(2))  # 0.3 for safety
        else:
            dt = self.time_step

        grid.setTime(int(self.simulation_duration / dt), dt)

        # Create medium with spatially varying properties
        medium = self._create_medium(maze)

        # Create source
        source = self._create_source(grid, source_pos)

        # Create sensor array
        sensor = self._create_sensor_array(grid, agent_pos)

        # Configure simulation options
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.pml_size,
            data_cast='single',
            save_to_disk=False,
            smooth_p0=False,
        )

        # Suppress warnings during simulation
        with warnings.catch_warnings():
            if not verbose:
                warnings.simplefilter("ignore")

            # Run simulation
            sensor_data = kspaceFirstOrder2D(
                medium=medium,
                kgrid=grid,
                source=source,
                sensor=sensor,
                simulation_options=simulation_options,
                execution_options={'verbose': verbose}
            )

        # Extract pressure data
        # sensor_data.p has shape (num_sensors, num_time_steps)
        pressure_data = sensor_data['p']

        # Ensure 2D array even if single sensor
        if pressure_data.ndim == 1:
            pressure_data = pressure_data.reshape(1, -1)

        return pressure_data.astype(np.float32)

    def compute_spectrogram(
        self,
        sensor_data: np.ndarray,
        nperseg: int = 64,
        noverlap: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert time-series sensor data to spectrograms for CNN input.

        Spectrograms provide a time-frequency representation that captures
        both temporal dynamics and frequency content of the acoustic signal.

        Parameters
        ----------
        sensor_data : np.ndarray
            Time-series data with shape (num_microphones, num_time_steps)
        nperseg : int, optional
            Length of each segment for STFT (default: 64)
        noverlap : int, optional
            Number of points to overlap between segments (default: nperseg // 2)

        Returns
        -------
        np.ndarray
            Spectrogram with shape (num_microphones, freq_bins, time_bins)
            Values are in log scale (dB) for better dynamic range.
        """
        from scipy import signal

        if noverlap is None:
            noverlap = nperseg // 2

        num_mics = sensor_data.shape[0]
        spectrograms = []

        for mic_idx in range(num_mics):
            # Compute Short-Time Fourier Transform
            f, t, Sxx = signal.spectrogram(
                sensor_data[mic_idx],
                fs=1.0 / self.time_step if self.time_step else 1.0,
                nperseg=nperseg,
                noverlap=noverlap,
                mode='magnitude'
            )

            # Convert to log scale (dB)
            # Add small epsilon to avoid log(0)
            Sxx_db = 20 * np.log10(Sxx + 1e-10)

            spectrograms.append(Sxx_db)

        # Stack along microphone dimension
        spectrogram_array = np.stack(spectrograms, axis=0)

        return spectrogram_array.astype(np.float32)
