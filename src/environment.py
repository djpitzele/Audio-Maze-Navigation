"""
Maze Environment and Navigation Oracle

This module provides maze generation algorithms and an A* pathfinding oracle
for providing optimal navigation labels during supervised learning.
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from enum import IntEnum
import heapq


class Action(IntEnum):
    """
    Discrete navigation actions.

    The agent can move in four cardinal directions or stop when reaching the goal.
    """
    STOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class MazeGenerator:
    """
    Generate 2D maze environments using various algorithms.

    This class provides multiple maze generation strategies suitable for
    acoustic navigation research, including simple random mazes, DFS-based
    mazes, and procedural layouts.

    Parameters
    ----------
    width : int
        Maze width in grid cells
    height : int
        Maze height in grid cells
    random_seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    width : int
        Maze width
    height : int
        Maze height
    rng : np.random.Generator
        Random number generator
    """

    def __init__(self, width: int, height: int, random_seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(random_seed)

    def generate_simple_maze(
        self,
        wall_probability: float = 0.3,
        ensure_borders: bool = True
    ) -> np.ndarray:
        """
        Generate a simple random maze.

        Creates a maze by randomly placing walls with a specified probability.
        Optionally adds solid borders around the perimeter.

        Parameters
        ----------
        wall_probability : float, optional
            Probability of each cell being a wall (default: 0.3)
        ensure_borders : bool, optional
            If True, create solid walls on maze perimeter (default: True)

        Returns
        -------
        np.ndarray
            Binary maze array where 0 = air, 1 = wall
        """
        # Create random maze
        maze = (self.rng.random((self.height, self.width)) < wall_probability).astype(np.int32)

        # Add border walls
        if ensure_borders:
            maze[0, :] = 1  # Top
            maze[-1, :] = 1  # Bottom
            maze[:, 0] = 1  # Left
            maze[:, -1] = 1  # Right

        return maze

    def generate_dfs_maze(self) -> np.ndarray:
        """
        Generate a maze using Depth-First Search (DFS) algorithm.

        This creates a perfect maze (no loops, single solution) using
        recursive backtracking. The maze has guaranteed connectivity
        between all open cells.

        Returns
        -------
        np.ndarray
            Binary maze array where 0 = air, 1 = wall

        Notes
        -----
        The algorithm works on cells separated by walls, so the output
        dimensions will be (2*height + 1, 2*width + 1) to accommodate
        walls between cells.
        """
        # Initialize maze with all walls
        maze_h = 2 * self.height + 1
        maze_w = 2 * self.width + 1
        maze = np.ones((maze_h, maze_w), dtype=np.int32)

        # Starting position (in cell coordinates)
        start_x, start_y = 0, 0

        # Convert to maze coordinates (cells are at odd indices)
        x, y = 2 * start_x + 1, 2 * start_y + 1
        maze[y, x] = 0

        # Stack for DFS
        stack = [(x, y)]
        visited = {(x, y)}

        # Define directions (up, down, left, right)
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            x, y = stack[-1]

            # Find unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Check bounds (stay within cell grid)
                if 1 <= nx < maze_w - 1 and 1 <= ny < maze_h - 1:
                    if (nx, ny) not in visited:
                        neighbors.append((nx, ny, dx, dy))

            if neighbors:
                # Choose random unvisited neighbor
                idx = self.rng.integers(0, len(neighbors))
                nx, ny, dx, dy = neighbors[idx]

                # Remove wall between current and neighbor
                wall_x = x + dx // 2
                wall_y = y + dy // 2
                maze[wall_y, wall_x] = 0

                # Mark neighbor as visited
                maze[ny, nx] = 0
                visited.add((nx, ny))

                # Add to stack
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()

        return maze

    def generate_rooms_and_corridors(
        self,
        num_rooms: int = 4,
        min_room_size: int = 3,
        max_room_size: int = 6
    ) -> np.ndarray:
        """
        Generate a maze with rooms connected by corridors.

        This creates a more structured environment with distinct acoustic
        signatures for rooms versus corridors.

        Parameters
        ----------
        num_rooms : int, optional
            Number of rooms to generate (default: 4)
        min_room_size : int, optional
            Minimum room dimension (default: 3)
        max_room_size : int, optional
            Maximum room dimension (default: 6)

        Returns
        -------
        np.ndarray
            Binary maze array where 0 = air, 1 = wall
        """
        # Start with all walls
        maze = np.ones((self.height, self.width), dtype=np.int32)

        rooms = []

        # Place rooms
        for _ in range(num_rooms):
            # Random room size
            room_h = self.rng.integers(min_room_size, max_room_size + 1)
            room_w = self.rng.integers(min_room_size, max_room_size + 1)

            # Random position (ensure within bounds)
            if room_h >= self.height or room_w >= self.width:
                continue

            y = self.rng.integers(1, max(2, self.height - room_h))
            x = self.rng.integers(1, max(2, self.width - room_w))

            # Carve out room
            maze[y:y + room_h, x:x + room_w] = 0

            # Store room center
            rooms.append((y + room_h // 2, x + room_w // 2))

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            y1, x1 = rooms[i]
            y2, x2 = rooms[i + 1]

            # Create L-shaped corridor
            # Horizontal then vertical
            maze[y1, min(x1, x2):max(x1, x2) + 1] = 0
            maze[min(y1, y2):max(y1, y2) + 1, x2] = 0

        return maze

    def get_random_walkable_position(self, maze: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Get a random walkable (air) position in the maze.

        Parameters
        ----------
        maze : np.ndarray
            Binary maze array

        Returns
        -------
        Optional[Tuple[int, int]]
            Random (row, col) position where maze value is 0, or None if no walkable cells
        """
        walkable = np.argwhere(maze == 0)

        if len(walkable) == 0:
            return None

        idx = self.rng.integers(0, len(walkable))
        return tuple(walkable[idx])

    def get_all_walkable_positions(self, maze: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all walkable positions in the maze.

        Parameters
        ----------
        maze : np.ndarray
            Binary maze array

        Returns
        -------
        List[Tuple[int, int]]
            List of all (row, col) positions where maze value is 0
        """
        walkable = np.argwhere(maze == 0)
        return [tuple(pos) for pos in walkable]


class Oracle:
    """
    A* pathfinding oracle for providing optimal navigation actions.

    This oracle computes shortest paths using the A* algorithm and provides
    the optimal next action for supervised learning. The oracle serves as
    the "teacher" during training.

    Parameters
    ----------
    maze : np.ndarray
        Binary maze array where 0 = air, 1 = wall
    """

    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Manhattan distance heuristic for A*.

        Parameters
        ----------
        pos1 : Tuple[int, int]
            First position (row, col)
        pos2 : Tuple[int, int]
            Second position (row, col)

        Returns
        -------
        float
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Action]]:
        """
        Get valid neighboring positions and corresponding actions.

        Parameters
        ----------
        pos : Tuple[int, int]
            Current position (row, col)

        Returns
        -------
        List[Tuple[Tuple[int, int], Action]]
            List of (neighbor_position, action) pairs
        """
        row, col = pos
        neighbors = []

        # Define movements: (delta_row, delta_col, action)
        moves = [
            (-1, 0, Action.UP),
            (1, 0, Action.DOWN),
            (0, -1, Action.LEFT),
            (0, 1, Action.RIGHT),
        ]

        for dr, dc, action in moves:
            new_row, new_col = row + dr, col + dc

            # Check bounds
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                # Check if walkable
                if self.maze[new_row, new_col] == 0:
                    neighbors.append(((new_row, new_col), action))

        return neighbors

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal using A*.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting position (row, col)
        goal : Tuple[int, int]
            Goal position (row, col)

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of positions from start to goal, or None if no path exists
        """
        # Priority queue: (f_score, counter, position, path)
        counter = 0
        heap = [(0, counter, start, [start])]
        visited: Set[Tuple[int, int]] = set()

        while heap:
            f_score, _, current, path = heapq.heappop(heap)

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)

            # Explore neighbors
            for neighbor, action in self._get_neighbors(current):
                if neighbor not in visited:
                    g_score = len(path)  # Cost so far
                    h_score = self._heuristic(neighbor, goal)
                    f_score = g_score + h_score

                    counter += 1
                    heapq.heappush(
                        heap,
                        (f_score, counter, neighbor, path + [neighbor])
                    )

        return None  # No path found

    def get_optimal_action(
        self,
        current_pos: Tuple[int, int],
        goal_pos: Tuple[int, int]
    ) -> Action:
        """
        Get the optimal next action from current position toward goal.

        This is the main interface for supervised learning. It returns the
        action that the agent should take to follow the shortest path.

        Parameters
        ----------
        current_pos : Tuple[int, int]
            Current position (row, col)
        goal_pos : Tuple[int, int]
            Goal position (row, col)

        Returns
        -------
        Action
            Optimal action to take (STOP if at goal or no path exists)

        Notes
        -----
        If the agent is already at the goal, returns Action.STOP.
        If no path exists, returns Action.STOP as a safe fallback.
        """
        # Check if already at goal
        if current_pos == goal_pos:
            return Action.STOP

        # Find optimal path
        path = self.find_path(current_pos, goal_pos)

        if path is None or len(path) < 2:
            # No path found or invalid path
            return Action.STOP

        # Next position in optimal path
        next_pos = path[1]

        # Determine action based on position difference
        dr = next_pos[0] - current_pos[0]
        dc = next_pos[1] - current_pos[1]

        if dr == -1:
            return Action.UP
        elif dr == 1:
            return Action.DOWN
        elif dc == -1:
            return Action.LEFT
        elif dc == 1:
            return Action.RIGHT
        else:
            # Shouldn't happen with valid path
            return Action.STOP

    def is_reachable(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> bool:
        """
        Check if goal is reachable from start.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting position
        goal : Tuple[int, int]
            Goal position

        Returns
        -------
        bool
            True if a path exists, False otherwise
        """
        return self.find_path(start, goal) is not None
