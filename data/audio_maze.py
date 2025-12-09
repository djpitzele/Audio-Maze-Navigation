from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.ShortestPaths import ShortestPaths
import matplotlib.pyplot as plt
from collections import deque

class AudioMaze:
    def __init__(self, width, height):
        self.gen_maze_obj(width, height)
        self.solve_maze()
        
    def gen_maze_obj(self, width, height):
        self.maze = Maze()
        
        self.maze.generator = Prims(width, height)
        self.maze.generate()
        
        self.maze.start = (1, 1)
        self.maze.end = (height * 2 - 1, width * 2 - 1)
        
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