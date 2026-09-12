"""
lidar_simulator.py
--------------------
Simulates a 360-degree LiDAR scan from the robot's current cell and uses
it to update a 2D Occupancy Grid — the exact "Process simulated LiDAR
sensor data to build a 2D Occupancy Grid" requirement from the brief.

Occupancy grid cell values:
    -1 = unknown / unexplored (robot has never scanned this far)
     0 = known free space
     1 = known occupied (wall or obstacle)

This is a simplified but faithful SLAM-style occupancy grid: every ray
marks the cells it passes through as FREE, and the cell where it stops
(a wall/obstacle) as OCCUPIED. Cells beyond a hit, or beyond max range,
stay UNKNOWN until a future scan reveals them.
"""

import math

import numpy as np

N_RAYS = 32          # angular resolution of the simulated LiDAR
MAX_RANGE = 6        # sensor range, in grid cells
STEP = 0.5           # ray marching step size, in grid cells


def create_unknown_grid(size):
    return np.full((size, size), -1, dtype=np.int8)


def simulate_lidar_scan(true_map, robot_pos):
    """
    Casts N_RAYS rays outward from robot_pos, up to MAX_RANGE cells,
    stepping through the TRUE map (ground truth) to see what the sensor
    would actually detect.

    Returns a list of (row, col, occupied: bool) for every cell the scan
    touched — free cells along the way, and the occupied cell that
    stopped the ray (if any).
    """
    hits = []
    r0, c0 = robot_pos

    for i in range(N_RAYS):
        angle = 2 * math.pi * i / N_RAYS
        dr, dc = math.sin(angle), math.cos(angle)

        dist = 0.0
        while dist <= MAX_RANGE:
            r = int(round(r0 + dr * dist))
            c = int(round(c0 + dc * dist))

            if r < 0 or r >= true_map.shape[0] or c < 0 or c >= true_map.shape[1]:
                break

            if true_map[r, c] == 1:
                hits.append((r, c, True))
                break
            else:
                hits.append((r, c, False))

            dist += STEP

    return hits


def update_occupancy_grid(occupancy_grid, hits):
    """Merges one LiDAR scan's hits into the persistent occupancy grid.
    Occupied always wins (a wall is a wall), free only overwrites unknown."""
    for r, c, occupied in hits:
        if occupied:
            occupancy_grid[r, c] = 1
        elif occupancy_grid[r, c] != 1:
            occupancy_grid[r, c] = 0
    return occupancy_grid
