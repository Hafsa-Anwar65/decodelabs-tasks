"""
maze_environment.py
--------------------
Defines the GROUND-TRUTH world the robot does NOT get to see directly.
The robot only ever learns about this world through simulated LiDAR scans
(see lidar_simulator.py) — exactly like a real AMR never has the factory
floor's true blueprint, only what its sensors have swept so far.

Grid convention:
    0 = free space
    1 = wall / obstacle

Coordinates are (row, col), row increasing downward, col increasing right.
"""

import numpy as np

GRID_SIZE = 12  # 12x12 grid (10x10 usable interior + border walls)

START = (1, 1)
GOAL = (10, 10)

# A dynamic obstacle "appears" mid-mission at this cell, once the robot
# reaches DYNAMIC_OBSTACLE_TRIGGER_STEP steps into its journey. This
# simulates something a static map could never predict — a dropped box,
# another robot, a person walking through the aisle.
DYNAMIC_OBSTACLE_CELL = (5, 6)
DYNAMIC_OBSTACLE_TRIGGER_STEP = 4


def build_true_map():
    """
    Builds the ground-truth occupancy map:
    - Outer border walls (the room's boundary)
    - An interior dividing wall at col=6, with TWO gaps (row 5 and row 9)
      so a genuine alternate route always exists if one gap gets blocked.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Border walls
    grid[0, :] = 1
    grid[GRID_SIZE - 1, :] = 1
    grid[:, 0] = 1
    grid[:, GRID_SIZE - 1] = 1

    # Interior dividing wall at column 6, rows 1..10, with two gaps
    wall_col = 6
    for r in range(1, GRID_SIZE - 1):
        if r in (5, 9):  # gaps / doorways
            continue
        grid[r, wall_col] = 1

    return grid


def is_free(true_map, row, col):
    if row < 0 or row >= true_map.shape[0] or col < 0 or col >= true_map.shape[1]:
        return False
    return true_map[row, col] == 0


def apply_dynamic_obstacle(true_map):
    """Mutates the ground-truth map to add the scripted dynamic obstacle.
    Called once, at DYNAMIC_OBSTACLE_TRIGGER_STEP, to simulate something
    unexpectedly appearing in the world."""
    r, c = DYNAMIC_OBSTACLE_CELL
    true_map[r, c] = 1
    return true_map
