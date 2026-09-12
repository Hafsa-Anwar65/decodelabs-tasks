"""
robot_navigation_sim.py
--------------------------
DecodeLabs Project 3 — Autonomous Mobile Robot (AMR) Navigation
Main simulation loop.

Ties together all three key requirements from the brief in one run:
  1. Simulated LiDAR -> 2D Occupancy Grid   (lidar_simulator.py)
  2. A* pathfinding for the shortest route   (astar_pathfinder.py)
  3. Dynamic obstacle avoidance / re-routing (this file's main loop)

Run:
    python3 robot_navigation_sim.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import maze_environment as env
from astar_pathfinder import a_star
from lidar_simulator import (
    MAX_RANGE,
    create_unknown_grid,
    simulate_lidar_scan,
    update_occupancy_grid,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_STEPS = 200


def move_towards(current, next_cell):
    """Single-cell step from current to next_cell (they must be adjacent)."""
    return next_cell


def run_simulation():
    true_map = env.build_true_map()
    occupancy_grid = create_unknown_grid(env.GRID_SIZE)

    robot_pos = env.START
    goal = env.GOAL

    trajectory = [robot_pos]
    replan_events = []
    dynamic_obstacle_spawned = False

    print(f"Mission: navigate from {env.START} to {goal}")
    print(f"Grid size: {env.GRID_SIZE}x{env.GRID_SIZE}\n")

    # --- Initial scan + plan ---
    hits = simulate_lidar_scan(true_map, robot_pos)
    occupancy_grid = update_occupancy_grid(occupancy_grid, hits)
    path = a_star(occupancy_grid, robot_pos, goal)

    if path is None:
        print("No initial path found even with optimistic unknown-space assumption. Aborting.")
        return

    print(f"Step 0: initial LiDAR scan complete. Planned path length = {len(path)} cells.")

    step = 0
    path_index = 1  # path[0] is the current position

    while robot_pos != goal and step < MAX_STEPS:
        step += 1

        # --- Scripted dynamic obstacle event ---
        if not dynamic_obstacle_spawned and step == env.DYNAMIC_OBSTACLE_TRIGGER_STEP:
            env.apply_dynamic_obstacle(true_map)
            dynamic_obstacle_spawned = True
            print(f"Step {step}: ⚠️  DYNAMIC OBSTACLE appeared at {env.DYNAMIC_OBSTACLE_CELL}!")

        # --- Move one cell along the current plan ---
        if path_index >= len(path):
            print(f"Step {step}: reached end of planned path without reaching goal — replanning.")
            path = None
        else:
            next_cell = path[path_index]

            # Re-scan before every move — a real AMR is always sensing.
            hits = simulate_lidar_scan(true_map, robot_pos)
            occupancy_grid = update_occupancy_grid(occupancy_grid, hits)

            if occupancy_grid[next_cell] == 1:
                # The path we were following is now blocked — the robot
                # "sees" this via LiDAR and must dynamically re-route.
                print(f"Step {step}: 🛑 Obstacle detected at {next_cell} — path blocked! Replanning...")
                path = a_star(occupancy_grid, robot_pos, goal)
                replan_events.append((step, robot_pos))

                if path is None:
                    print(f"Step {step}: ❌ No alternate route found with current knowledge. Robot stops.")
                    break

                path_index = 1
                print(f"Step {step}: ✅ New path found, length = {len(path)} cells.")
                continue

            robot_pos = move_towards(robot_pos, next_cell)
            trajectory.append(robot_pos)
            path_index += 1

    if robot_pos == goal:
        print(f"\n🎉 SUCCESS: robot reached goal {goal} in {step} steps ({len(trajectory)} cells traveled).")
    else:
        print(f"\n❌ FAILED to reach goal within {MAX_STEPS} steps.")

    visualize(true_map, occupancy_grid, trajectory, replan_events)
    return trajectory, replan_events


def visualize(true_map, occupancy_grid, trajectory, replan_events):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Left: ground truth (for grading / comparison only — the robot
    #     never sees this directly) ---
    ax = axes[0]
    ax.imshow(true_map, cmap="Greys", vmin=0, vmax=1)
    ax.set_title("Ground Truth Map (hidden from robot)")
    ax.plot(env.START[1], env.START[0], "go", markersize=12, label="Start")
    ax.plot(env.GOAL[1], env.GOAL[0], "b*", markersize=16, label="Goal")
    ax.plot(*env.DYNAMIC_OBSTACLE_CELL[::-1], "rx", markersize=12, mew=3, label="Dynamic obstacle")
    ax.legend(loc="upper left", fontsize=8)

    # --- Right: what the robot actually built + did ---
    ax = axes[1]
    display_grid = np.where(occupancy_grid == -1, 0.5, occupancy_grid)
    ax.imshow(display_grid, cmap="Greys", vmin=0, vmax=1)
    ax.set_title("Robot's Occupancy Grid + Trajectory")

    traj = np.array(trajectory)
    ax.plot(traj[:, 1], traj[:, 0], "b-", linewidth=2, label="Actual path taken")
    ax.plot(env.START[1], env.START[0], "go", markersize=12, label="Start")
    ax.plot(env.GOAL[1], env.GOAL[0], "b*", markersize=16, label="Goal")

    for i, (step, pos) in enumerate(replan_events):
        ax.plot(pos[1], pos[0], "r^", markersize=12, label="Replan point" if i == 0 else None)

    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "navigation_result.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nVisualization saved to: {out_path}")


if __name__ == "__main__":
    run_simulation()
