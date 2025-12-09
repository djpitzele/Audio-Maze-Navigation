from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.ShortestPaths import ShortestPaths
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from scipy.ndimage import binary_dilation

class AudioMaze:
    def __init__(self, width, height, corridor_width=3, use_cave_style=False):
        """
        Generate maze with wider corridors suitable for microphone arrays.

        Args:
            width: Base maze width (will be scaled up)
            height: Base maze height (will be scaled up)
            corridor_width: Minimum corridor width in grid cells (default: 3 for 8-mic array)
            use_cave_style: If True, create more open cave-like spaces
        """
        self.corridor_width = corridor_width
        self.use_cave_style = use_cave_style
        self.gen_maze_obj(width, height)
        self.solve_maze()

    def gen_maze_obj(self, width, height):
        self.maze = Maze()

        self.maze.generator = Prims(width, height)
        self.maze.generate()

        # Store original maze before widening
        original_grid = self.maze.grid.copy()
        original_start = (1, 1)
        original_end = (height * 2 - 1, width * 2 - 1)

        # Widen corridors by dilating the path (eroding walls)
        if self.corridor_width > 1:
            self.maze.grid = self._widen_corridors(original_grid, self.corridor_width)

        # Apply cave-style modifications if requested
        if self.use_cave_style:
            self.maze.grid = self._make_cave_like(self.maze.grid)

        # Update start/end positions (scale to new grid)
        rows, cols = self.maze.grid.shape
        # Find suitable start/end in widened maze
        air_cells = np.argwhere(self.maze.grid == 0)
        if len(air_cells) > 0:
            self.maze.start = tuple(air_cells[0])  # First air cell
            self.maze.end = tuple(air_cells[-1])   # Last air cell
        else:
            self.maze.start = (1, 1)
            self.maze.end = (rows - 2, cols - 2)

    def _widen_corridors(self, grid, width):
        """
        Widen corridors by eroding walls (dilating paths).

        Args:
            grid: Binary maze grid (0=path, 1=wall)
            width: Target corridor width

        Returns:
            Widened maze grid
        """
        # Create path mask (inverse of grid)
        path_mask = (grid == 0)

        # Dilate paths (erode walls) to make corridors wider
        # Use circular structuring element for natural widening
        iterations = (width - 1) // 2
        structure = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=bool)

        widened_paths = binary_dilation(path_mask, structure=structure, iterations=iterations)

        # Convert back to maze format (0=path, 1=wall)
        widened_grid = (~widened_paths).astype(np.int32)

        # Ensure borders are walls
        widened_grid[0, :] = 1
        widened_grid[-1, :] = 1
        widened_grid[:, 0] = 1
        widened_grid[:, -1] = 1

        return widened_grid

    def _make_cave_like(self, grid):
        """
        Make maze more cave-like with varied open spaces.
        Randomly removes some walls to create larger chambers.

        Args:
            grid: Binary maze grid

        Returns:
            Modified cave-like maze grid
        """
        cave_grid = grid.copy()
        rows, cols = cave_grid.shape

        # Create random large open areas (chambers)
        num_chambers = np.random.randint(2, 5)
        for _ in range(num_chambers):
            # Random chamber position and size
            chamber_size = np.random.randint(5, 10)
            center_r = np.random.randint(chamber_size, rows - chamber_size)
            center_c = np.random.randint(chamber_size, cols - chamber_size)

            # Create circular chamber
            for r in range(max(1, center_r - chamber_size), min(rows - 1, center_r + chamber_size)):
                for c in range(max(1, center_c - chamber_size), min(cols - 1, center_c + chamber_size)):
                    dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                    if dist < chamber_size:
                        cave_grid[r, c] = 0  # Make it a path

        return cave_grid
        
    def solve_maze(self):
        self.maze.solver = ShortestPaths()
        self.maze.solve()

        rows, cols = self.maze.grid.shape
        self.action_grid = [["" for _ in range(cols)] for _ in range(rows)]
        start_node = self.maze.end
        self.action_grid[start_node[0]][start_node[1]] = "stop"
        
        queue = deque([start_node]) # for BFS
        visited = {start_node}
        
        # Each tuple: ((row_offset, col_offset), action_at_neighbor)
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
                    if self.maze.grid[nr, nc] == 0 and (nr, nc) not in visited:
                        self.action_grid[nr][nc] = action
                        visited.add((nr, nc))
                        queue.append((nr, nc))
    
    def visualize_maze(self):
        to_show = self.maze.grid.astype(float)
        to_show[self.maze.start[0], self.maze.start[1]] = 0.5  # Mark start
        to_show[self.maze.end[0], self.maze.end[1]] = 0.75    # Mark end
        plt.imshow(to_show, cmap='Greys')
        plt.show()
        
    def visualize_solution(self):
        symbol_map = {
            "up": "^",
            "down": "v",
            "left": "<",
            "right": ">",
            "stop": "*",
            "": "#" # wall
        }
        print("\n--- Maze Flow Field ---")
        for row in self.action_grid:
            line_str = []
            for action in row:
                symbol = symbol_map.get(action, "?")
                line_str.append(f" {symbol} ")
            print("".join(line_str))
    
if __name__ == "__main__":
    am = AudioMaze(15, 15)
    am.visualize_maze()
    am.visualize_solution()