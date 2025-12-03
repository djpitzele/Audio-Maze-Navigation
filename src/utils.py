"""
Visualization and utility functions for acoustic navigation research.

This module provides tools for visualizing mazes, spectrograms, training
progress, and agent navigation trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Optional, Union
import torch


def plot_maze(
    maze: np.ndarray,
    agent_pos: Optional[Tuple[int, int]] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
    source_pos: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    title: str = "Maze Environment",
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Visualize a maze with optional agent, goal, and path overlay.

    Parameters
    ----------
    maze : np.ndarray
        Binary maze array (0=air, 1=wall)
    agent_pos : Tuple[int, int], optional
        Agent position (row, col)
    goal_pos : Tuple[int, int], optional
        Goal position (row, col)
    source_pos : Tuple[int, int], optional
        Sound source position (row, col)
    path : List[Tuple[int, int]], optional
        List of positions forming a path
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot maze (walls in black, air in white)
    cmap = LinearSegmentedColormap.from_list('maze', ['white', 'black'])
    ax.imshow(maze, cmap=cmap, interpolation='nearest')

    # Plot path if provided
    if path is not None and len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2, alpha=0.6, label='Path')

    # Plot source position
    if source_pos is not None:
        ax.add_patch(Circle(
            (source_pos[1], source_pos[0]),
            radius=0.4,
            color='orange',
            alpha=0.8,
            label='Source'
        ))

    # Plot agent position
    if agent_pos is not None:
        ax.add_patch(Circle(
            (agent_pos[1], agent_pos[0]),
            radius=0.4,
            color='blue',
            alpha=0.8,
            label='Agent'
        ))

    # Plot goal position
    if goal_pos is not None:
        ax.add_patch(Circle(
            (goal_pos[1], goal_pos[0]),
            radius=0.4,
            color='green',
            alpha=0.8,
            label='Goal'
        ))

    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, alpha=0.3)

    # Add legend if any markers are present
    if agent_pos is not None or goal_pos is not None or source_pos is not None or path is not None:
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_spectrogram(
    spectrogram: Union[np.ndarray, torch.Tensor],
    microphone_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Visualize a spectrogram from a single microphone.

    Parameters
    ----------
    spectrogram : np.ndarray or torch.Tensor
        Spectrogram with shape (num_mics, freq_bins, time_bins) or (freq_bins, time_bins)
    microphone_idx : int, optional
        Which microphone to visualize if multi-channel (default: 0)
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Convert to numpy if torch tensor
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()

    # Extract single microphone if multi-channel
    if spectrogram.ndim == 3:
        spec_2d = spectrogram[microphone_idx]
    else:
        spec_2d = spectrogram

    # Plot
    im = ax.imshow(
        spec_2d,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )

    if title is None:
        title = f'Spectrogram (Microphone {microphone_idx})'

    ax.set_title(title)
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Frequency Bin')

    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    plt.tight_layout()

    return fig


def plot_multi_channel_spectrogram(
    spectrogram: Union[np.ndarray, torch.Tensor],
    title: str = "Multi-Channel Spectrogram",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Visualize spectrograms from all microphones.

    Parameters
    ----------
    spectrogram : np.ndarray or torch.Tensor
        Spectrogram with shape (num_mics, freq_bins, time_bins)
    title : str, optional
        Overall title
    figsize : Tuple[int, int], optional
        Figure size (auto-calculated if None)

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Convert to numpy if torch tensor
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()

    num_mics = spectrogram.shape[0]

    # Calculate grid layout
    ncols = min(4, num_mics)
    nrows = (num_mics + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if num_mics == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for mic_idx in range(num_mics):
        ax = axes[mic_idx]
        spec_2d = spectrogram[mic_idx]

        im = ax.imshow(
            spec_2d,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )

        ax.set_title(f'Mic {mic_idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(num_mics, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_sensor_data(
    sensor_data: np.ndarray,
    sample_rate: float = 1.0,
    title: str = "Sensor Time Series",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot raw sensor time-series data from multiple microphones.

    Parameters
    ----------
    sensor_data : np.ndarray
        Time-series data with shape (num_mics, num_time_steps)
    sample_rate : float, optional
        Sampling rate for x-axis (default: 1.0)
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    num_mics, num_time_steps = sensor_data.shape
    time_axis = np.arange(num_time_steps) / sample_rate

    fig, axes = plt.subplots(num_mics, 1, figsize=figsize, sharex=True)

    if num_mics == 1:
        axes = [axes]

    for mic_idx in range(num_mics):
        axes[mic_idx].plot(time_axis, sensor_data[mic_idx], linewidth=0.5)
        axes[mic_idx].set_ylabel(f'Mic {mic_idx}')
        axes[mic_idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot training and validation metrics.

    Parameters
    ----------
    train_losses : List[float]
        Training loss per epoch
    val_losses : List[float], optional
        Validation loss per epoch
    train_accuracies : List[float], optional
        Training accuracy per epoch
    val_accuracies : List[float], optional
        Validation accuracy per epoch
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    has_accuracy = train_accuracies is not None

    if has_accuracy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))

    epochs = np.arange(1, len(train_losses) + 1)

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    if has_accuracy:
        ax2.plot(epochs, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
        if val_accuracies is not None:
            ax2.plot(epochs, val_accuracies, 'r-', label='Val Accuracy', linewidth=2)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_navigation_episode(
    maze: np.ndarray,
    trajectory: List[Tuple[int, int]],
    goal_pos: Tuple[int, int],
    actions: List[int],
    title: str = "Navigation Episode",
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Visualize a complete navigation episode with actions.

    Parameters
    ----------
    maze : np.ndarray
        Binary maze array
    trajectory : List[Tuple[int, int]]
        List of positions visited by the agent
    goal_pos : Tuple[int, int]
        Goal position
    actions : List[int]
        List of actions taken at each step
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot maze
    cmap = LinearSegmentedColormap.from_list('maze', ['white', 'black'])
    ax.imshow(maze, cmap=cmap, interpolation='nearest', alpha=0.7)

    # Plot trajectory
    if len(trajectory) > 0:
        trajectory_array = np.array(trajectory)
        ax.plot(
            trajectory_array[:, 1],
            trajectory_array[:, 0],
            'b-',
            linewidth=2,
            alpha=0.6,
            label='Trajectory'
        )

        # Mark start and end
        start_pos = trajectory[0]
        end_pos = trajectory[-1]

        ax.add_patch(Circle(
            (start_pos[1], start_pos[0]),
            radius=0.5,
            color='blue',
            alpha=0.8,
            label='Start'
        ))

        ax.add_patch(Circle(
            (end_pos[1], end_pos[0]),
            radius=0.5,
            color='cyan',
            alpha=0.8,
            label='End'
        ))

    # Plot goal
    ax.add_patch(Circle(
        (goal_pos[1], goal_pos[0]),
        radius=0.5,
        color='green',
        alpha=0.8,
        label='Goal'
    ))

    # Add action annotations at key points
    action_names = ['STOP', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    sample_points = np.linspace(0, len(trajectory) - 1, min(10, len(trajectory)), dtype=int)

    for idx in sample_points:
        if idx < len(actions):
            pos = trajectory[idx]
            action = actions[idx]
            ax.annotate(
                action_names[action],
                xy=(pos[1], pos[0]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

    ax.set_title(f"{title}\nSteps: {len(trajectory)}")
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot confusion matrix for action predictions.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix (num_classes, num_classes)
    class_names : List[str], optional
        Names of action classes
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    normalize : bool, optional
        If True, normalize by row (true labels)

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if class_names is None:
        class_names = ['STOP', 'UP', 'DOWN', 'LEFT', 'RIGHT']

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Action',
        xlabel='Predicted Action'
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j, i,
                format(confusion_matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black"
            )

    plt.tight_layout()
    return fig


def create_animation_frames(
    maze: np.ndarray,
    trajectory: List[Tuple[int, int]],
    goal_pos: Tuple[int, int],
    output_dir: str = "animation_frames",
    figsize: Tuple[int, int] = (8, 8),
):
    """
    Create individual frames for animation of agent navigation.

    Parameters
    ----------
    maze : np.ndarray
        Binary maze array
    trajectory : List[Tuple[int, int]]
        Agent trajectory
    goal_pos : Tuple[int, int]
        Goal position
    output_dir : str, optional
        Directory to save frames
    figsize : Tuple[int, int], optional
        Figure size

    Notes
    -----
    Frames are saved as PNG files that can be combined into a video using
    external tools like ffmpeg:
    `ffmpeg -framerate 5 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4`
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list('maze', ['white', 'black'])

    for step, pos in enumerate(trajectory):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot maze
        ax.imshow(maze, cmap=cmap, interpolation='nearest', alpha=0.7)

        # Plot trajectory so far
        if step > 0:
            traj_so_far = np.array(trajectory[:step + 1])
            ax.plot(
                traj_so_far[:, 1],
                traj_so_far[:, 0],
                'b-',
                linewidth=2,
                alpha=0.5
            )

        # Plot current position
        ax.add_patch(Circle(
            (pos[1], pos[0]),
            radius=0.5,
            color='blue',
            alpha=0.8
        ))

        # Plot goal
        ax.add_patch(Circle(
            (goal_pos[1], goal_pos[0]),
            radius=0.5,
            color='green',
            alpha=0.8
        ))

        ax.set_title(f"Step {step}/{len(trajectory) - 1}")
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / f"frame_{step:04d}.png", dpi=100)
        plt.close(fig)

    print(f"Created {len(trajectory)} frames in {output_dir}/")


def get_microphone_positions(sensor_mask: np.ndarray) -> list:
    """
    Extract microphone positions from a sensor mask array.

    Parameters
    ----------
    sensor_mask : np.ndarray
        Boolean array where True indicates microphone positions

    Returns
    -------
    list
        List of (row, col) tuples for each microphone position
    """
    positions = []
    rows, cols = np.where(sensor_mask)
    for r, c in zip(rows, cols):
        positions.append((int(r), int(c)))
    return positions


def plot_acoustic_field(
    maze: np.ndarray,
    sound_speed: np.ndarray,
    agent_pos: Tuple[int, int],
    source_pos: Tuple[int, int],
    sensor_data: Optional[np.ndarray] = None,
    microphone_positions: Optional[list] = None,
    time_idx: int = 0,
    title: str = "Acoustic Environment",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Visualize the acoustic properties of the maze and sound propagation.

    Parameters
    ----------
    maze : np.ndarray
        Binary maze array (0=air, 1=wall)
    sound_speed : np.ndarray
        Sound speed field (m/s) at each grid point
    agent_pos : Tuple[int, int]
        Agent position (row, col)
    source_pos : Tuple[int, int]
        Sound source position (row, col)
    sensor_data : np.ndarray, optional
        Time-series sensor data to plot
    microphone_positions : list, optional
        List of (row, col) tuples for microphone positions
    time_idx : int, optional
        Which time index to display (default: 0)
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if sensor_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))

    # Plot 1: Sound speed field (shows maze geometry acoustically)
    im1 = ax1.imshow(
        sound_speed,
        cmap='viridis',
        interpolation='nearest',
        origin='lower'
    )
    ax1.set_title('Acoustic Properties (Sound Speed)')
    ax1.set_xlabel('X (grid points)')
    ax1.set_ylabel('Y (grid points)')

    # Add markers
    ax1.plot(source_pos[1], source_pos[0], 'r*', markersize=15, label='Source')
    ax1.plot(agent_pos[1], agent_pos[0], 'bo', markersize=10, label='Agent')

    # Add microphone positions if provided
    if microphone_positions is not None and len(microphone_positions) > 0:
        mic_rows = [pos[0] for pos in microphone_positions]
        mic_cols = [pos[1] for pos in microphone_positions]
        ax1.scatter(mic_cols, mic_rows, c='cyan', marker='x', s=100,
                   linewidths=2, label=f'Mics ({len(microphone_positions)})')

    ax1.legend()

    plt.colorbar(im1, ax=ax1, label='Sound Speed (m/s)')

    # Plot 2: Sensor signal (if provided)
    if sensor_data is not None:
        num_sensors = sensor_data.shape[0]
        time_axis = np.arange(sensor_data.shape[1])

        for i in range(num_sensors):
            ax2.plot(time_axis, sensor_data[i], label=f'Sensor {i}', alpha=0.7)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Pressure')
        ax2.set_title('Sensor Time Series')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
