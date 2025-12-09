"""
Test script to visualize different maze configurations:
1. Original narrow corridors
2. Widened corridors (corridor_width=3)
3. Cave-style with wider corridors
"""
import numpy as np
import matplotlib.pyplot as plt
from audio_maze import AudioMaze

# Generate three different maze styles
print("Generating mazes...")

# 1. Original narrow corridors (corridor_width=1)
maze_narrow = AudioMaze(10, 10, corridor_width=1, use_cave_style=False)
grid_narrow = maze_narrow.maze.grid

# 2. Wide corridors (corridor_width=3)
maze_wide = AudioMaze(10, 10, corridor_width=3, use_cave_style=False)
grid_wide = maze_wide.maze.grid

# 3. Cave style with wide corridors
maze_cave = AudioMaze(10, 10, corridor_width=3, use_cave_style=True)
grid_cave = maze_cave.maze.grid

# Print statistics
print(f"\n1. Narrow corridors:")
print(f"   Grid: {grid_narrow.shape}")
print(f"   Walls: {100*grid_narrow.mean():.1f}%")
print(f"   Air cells: {(grid_narrow == 0).sum()}")

print(f"\n2. Wide corridors (3 cells):")
print(f"   Grid: {grid_wide.shape}")
print(f"   Walls: {100*grid_wide.mean():.1f}%")
print(f"   Air cells: {(grid_wide == 0).sum()}")

print(f"\n3. Cave style:")
print(f"   Grid: {grid_cave.shape}")
print(f"   Walls: {100*grid_cave.mean():.1f}%")
print(f"   Air cells: {(grid_cave == 0).sum()}")

# Visualize all three
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(grid_narrow.T, origin='lower', cmap='binary')
axes[0].set_title(f'Narrow Corridors (1 cell)\n{grid_narrow.shape[0]}x{grid_narrow.shape[1]} grid')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].axis('image')

axes[1].imshow(grid_wide.T, origin='lower', cmap='binary')
axes[1].set_title(f'Wide Corridors (3 cells)\n{grid_wide.shape[0]}x{grid_wide.shape[1]} grid')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].axis('image')

axes[2].imshow(grid_cave.T, origin='lower', cmap='binary')
axes[2].set_title(f'Cave Style (3 cells + chambers)\n{grid_cave.shape[0]}x{grid_cave.shape[1]} grid')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].axis('image')

plt.tight_layout()
plt.savefig('../dataset/maze_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved comparison to ../dataset/maze_comparison.png")
plt.show()

# Demonstrate 8-mic array fits in wide corridor
print("\n" + "="*60)
print("Testing 8-mic array placement in wide corridor:")
print("="*60)

air_cells = np.argwhere(grid_wide == 0)
if len(air_cells) > 10:
    test_pos = air_cells[len(air_cells)//2]  # Middle position
    test_y, test_x = test_pos

    # Check if 3x3 neighborhood is all air
    valid_3x3 = True
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = test_y + dy, test_x + dx
            if ny < 0 or ny >= grid_wide.shape[0] or nx < 0 or nx >= grid_wide.shape[1]:
                valid_3x3 = False
            elif grid_wide[ny, nx] == 1:
                valid_3x3 = False

    print(f"Test position: ({test_y}, {test_x})")
    print(f"3x3 neighborhood clear: {valid_3x3}")

    if valid_3x3:
        print("✓ 8-mic array will fit!")
    else:
        print("⚠ Need to find better position or increase corridor_width")
