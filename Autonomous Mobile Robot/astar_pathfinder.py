"""
astar_pathfinder.py
---------------------
Classic A* search over the robot's occupancy grid (NOT the ground-truth
map — the robot only knows what it has scanned). This is the "implement
a basic pathfinding algorithm (like A* or Dijkstra's) to find the
shortest route" requirement from the brief.

Design choices, explained:
- UNKNOWN cells (-1) are treated as traversable. A real AMR usually
  assumes unexplored space is passable until proven otherwise — refusing
  to ever enter unknown territory would make exploration impossible.
- OCCUPIED cells (1) are always blocked.
- 4-connected movement (N/S/E/W) with Manhattan distance heuristic —
  admissible and consistent, guaranteeing the optimal path.
"""

import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos, grid_shape):
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (nr, nc) for nr, nc in candidates
        if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]
    ]


def a_star(occupancy_grid, start, goal):
    """
    Returns the shortest path (list of (row, col) tuples, start to goal
    inclusive) using A* search, or None if no path currently exists given
    what the robot has explored so far.
    """
    if occupancy_grid[goal] == 1:
        return None  # goal itself is known-blocked

    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in get_neighbors(current, occupancy_grid.shape):
            if occupancy_grid[neighbor] == 1:
                continue  # known obstacle — never step here

            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None  # no path found — goal unreachable given current knowledge


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
