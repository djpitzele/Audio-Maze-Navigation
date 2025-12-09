"""
Cave-like environment generator for acoustic navigation.
Creates irregular spaces with variable wall thickness and open pathways.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from collections import deque
import matplotlib.pyplot as plt


class AudioCave:
    def __init__(self, width, height, min_corridor_width=3, wall_thickness_range=(1, 8)):
        """
        Generate cave-like environment with variable wall thickness.

        Args:
            width: Environment width in grid cells
            height: Environment height in grid cells
            min_corridor_width: Minimum pathway width (default: 3 for 8-mic array)
            wall_thickness_range: (min, max) wall thickness in grid cells
        """
        self.width = width
        self.height = height
        self.min_corridor_width = min_corridor_width
        self.wall_thickness_range = wall_thickness_range

        self.gen_cave()
        self.solve_maze()

    def gen_cave(self):
        """Generate cave-like structure using cellular automata."""
        # Start with random noise
        np.random.seed()
        grid = np.random.rand(self.height, self.width) > 0.55  # ~45% walls initially

        # Apply cellular automata rules to create natural cave structure
        for iteration in range(5):
            grid = self._smooth_cave(grid, neighbors_threshold=5)

        # Ensure borders are walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        # Vary wall thickness
        grid = self._add_wall_thickness(grid)

        # Ensure minimum corridor width
        grid = self._ensure_min_corridor_width(grid, self.min_corridor_width)

        # Find largest connected component (main cave)
        grid = self._keep_largest_component(grid)

        # Find start and goal positions
        air_cells = np.argwhere(grid == 0)
        if len(air_cells) > 0:
            # Start at top-left area
            top_left = air_cells[np.argmin(air_cells[:, 0] + air_cells[:, 1])]
            # Goal at bottom-right area
            bottom_right = air_cells[np.argmax(air_cells[:, 0] + air_cells[:, 1])]
            self.start = tuple(top_left)
            self.end = tuple(bottom_right)
        else:
            self.start = (1, 1)
            self.end = (self.height - 2, self.width - 2)

        self.grid = grid.astype(np.int32)

    def _smooth_cave(self, grid, neighbors_threshold=5):
        """
        Apply cellular automata smoothing.
        Cell becomes wall if it has >= neighbors_threshold wall neighbors.
        """
        new_grid = grid.copy()
        rows, cols = grid.shape

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # Count wall neighbors (3x3 neighborhood)
                wall_count = grid[r-1:r+2, c-1:c+2].sum()

                if wall_count >= neighbors_threshold:
                    new_grid[r, c] = 1  # Become wall
                elif wall_count <= 3:
                    new_grid[r, c] = 0  # Become air

        return new_grid

    def _add_wall_thickness(self, grid):
        """Add variable wall thickness for realistic sound propagation."""
        wall_mask = (grid == 1)
        thickened = grid.copy()

        # Randomly thicken some walls
        num_thick_walls = int(wall_mask.sum() * 0.2)  # 20% of walls
        wall_positions = np.argwhere(wall_mask)

        if len(wall_positions) > 0:
            thick_wall_indices = np.random.choice(
                len(wall_positions),
                size=min(num_thick_walls, len(wall_positions)),
                replace=False
            )

            for idx in thick_wall_indices:
                r, c = wall_positions[idx]
                thickness = np.random.randint(*self.wall_thickness_range)

                # Expand wall in random direction
                if np.random.rand() > 0.5:  # Horizontal thickening
                    for dc in range(-thickness//2, thickness//2 + 1):
                        nc = c + dc
                        if 0 < nc < self.width - 1:
                            thickened[r, nc] = 1
                else:  # Vertical thickening
                    for dr in range(-thickness//2, thickness//2 + 1):
                        nr = r + dr
                        if 0 < nr < self.height - 1:
                            thickened[nr, c] = 1

        return thickened

    def _ensure_min_corridor_width(self, grid, min_width):
        """Ensure all passable corridors are at least min_width wide."""
        path_mask = (grid == 0)

        # Dilate paths to ensure minimum width
        iterations = (min_width - 1) // 2
        structure = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=bool)

        widened_paths = binary_dilation(path_mask, structure=structure, iterations=iterations)

        widened_grid = (~widened_paths).astype(np.int32)

        # Ensure borders remain walls
        widened_grid[0, :] = 1
        widened_grid[-1, :] = 1
        widened_grid[:, 0] = 1
        widened_grid[:, -1] = 1

        return widened_grid

    def _keep_largest_component(self, grid):
        """Keep only the largest connected air component (main cave)."""
        air_mask = (grid == 0)
        rows, cols = grid.shape

        visited = np.zeros_like(air_mask, dtype=bool)
        components = []

        def bfs(start_r, start_c):
            """Flood fill to find connected component."""
            component = []
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True

            while queue:
                r, c = queue.popleft()
                component.append((r, c))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        air_mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            return component

        # Find all connected components
        for r in range(rows):
            for c in range(cols):
                if air_mask[r, c] and not visited[r, c]:
                    component = bfs(r, c)
                    components.append(component)

        if not components:
            return grid

        # Keep largest component
        largest = max(components, key=len)
        new_grid = np.ones_like(grid)
        for r, c in largest:
            new_grid[r, c] = 0

        return new_grid

    def solve_maze(self):
        """Compute action labels using BFS from goal."""
        rows, cols = self.grid.shape
        self.action_grid = [["" for _ in range(cols)] for _ in range(rows)]

        start_node = self.end
        self.action_grid[start_node[0]][start_node[1]] = "stop"

        queue = deque([start_node])
        visited = {start_node}

        directions = [
            ((0, 1), "left"),
            ((0, -1), "right"),
            ((1, 0), "up"),
            ((-1, 0), "down")
        ]

        while queue:
            r, c = queue.popleft()
            for (dr, dc), action in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self.grid[nr, nc] == 0 and (nr, nc) not in visited:
                        self.action_grid[nr][nc] = action
                        visited.add((nr, nc))
                        queue.append((nr, nc))

    def visualize(self):
        """Visualize cave and action field."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Cave layout
        axes[0].imshow(self.grid.T, origin='lower', cmap='binary')
        axes[0].scatter([self.start[1]], [self.start[0]], s=200, c='green',
                       marker='o', edgecolors='black', linewidths=2, label='Start', zorder=10)
        axes[0].scatter([self.end[1]], [self.end[0]], s=200, c='red',
                       marker='*', edgecolors='black', linewidths=2, label='Goal', zorder=10)
        axes[0].set_title(f'Cave Layout ({self.height}x{self.width})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].legend()
        axes[0].axis('image')

        # Action field
        symbol_map = {"up": 1, "down": 2, "left": 3, "right": 4, "stop": 5, "": 0}
        action_numeric = np.vectorize(symbol_map.get)(self.action_grid)
        axes[1].imshow(action_numeric.T, origin='lower', cmap='tab10', vmin=0, vmax=5)
        axes[1].contour(self.grid.T, levels=[0.5], colors='black', linewidths=1)
        axes[1].scatter([self.end[1]], [self.end[0]], s=200, c='red',
                       marker='*', edgecolors='black', linewidths=2, label='Goal', zorder=10)
        axes[1].set_title('Action Field')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        axes[1].axis('image')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Generating cave environment...")
    cave = AudioCave(width=80, height=80, min_corridor_width=3, wall_thickness_range=(1, 8))

    print(f"Cave size: {cave.height}x{cave.width}")
    print(f"Walls: {100*cave.grid.mean():.1f}%")
    print(f"Air: {100*(1-cave.grid.mean()):.1f}%")
    print(f"Start: {cave.start}")
    print(f"Goal: {cave.end}")

    cave.visualize()
