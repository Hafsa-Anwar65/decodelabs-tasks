"""
robotic_arm_sim.py
--------------------
Main simulation loop for Project 1 -- Robotic Arm Kinematics & Path
Planning. Ties together forward_kinematics.py, inverse_kinematics.py and
trajectory_planner.py into the full IPO pipeline from the slide deck:

    Input Stage  (goal state)      -> Point A, Point B, obstacle
    Process Stage (the math)       -> quintic Cartesian path + collision
                                       check + per-waypoint IK
    Output Stage (the muscle)      -> joint-angle trajectory, ready to be
                                       handed to a FollowJointTrajectory
                                       action server on real/simulated
                                       hardware

Since this is a training-kit simulation (no physical ROS/Gazebo cluster
attached to this environment), the "RViz/Gazebo" visualization step is
reproduced with a matplotlib 3D rendering of the arm's link geometry at
several key frames plus the full end-effector path -- this shows exactly
what RViz would show (kinematic intent: does the arm reach the target
without colliding?).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from forward_kinematics import forward_kinematics, N_JOINTS
from inverse_kinematics import solve_ik
from trajectory_planner import plan_cartesian_trajectory

# ---------------------------------------------------------------------
# Mission definition (Input Stage)
# ---------------------------------------------------------------------
POINT_A = np.array([0.30, -0.25, 0.25])   # start XYZ (m)
POINT_B = np.array([0.30, 0.25, 0.25])    # goal XYZ (m)
OBSTACLE_CENTER = np.array([0.30, 0.0, 0.25])
OBSTACLE_RADIUS = 0.09                    # m
N_WAYPOINTS = 40


def solve_ik_along_trajectory(waypoints):
    """
    Solves IK at every Cartesian waypoint, warm-starting each solve from
    the previous waypoint's joint solution (keeps the arm from jumping
    between very different configurations frame-to-frame).

    Returns: joint_trajectory (n_steps, 6), all_converged (bool), log
    """
    joint_trajectory = np.zeros((len(waypoints), N_JOINTS))
    seed = None
    all_converged = True
    log = []

    for i, wp in enumerate(waypoints):
        q, converged, iters, err = solve_ik(wp, seed_angles=seed)
        joint_trajectory[i] = q
        seed = q
        if not converged:
            all_converged = False
            log.append(f"  Waypoint {i:02d}: IK FAILED to converge "
                       f"(residual={err * 1000:.2f} mm) at target {np.round(wp, 3)}")

    if all_converged:
        log.append(f"  All {len(waypoints)} waypoints solved within "
                   f"{1000 * 1e-3:.1f} mm tolerance.")
    return joint_trajectory, all_converged, log


def _arm_link_points(joint_angles):
    _, frames = forward_kinematics(joint_angles, return_all_frames=True)
    return np.array([f[:3, 3] for f in frames])


def plot_result(waypoints, joint_trajectory, replanned, out_path):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # End-effector Cartesian path actually taken
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
             color="#e0457b", linewidth=2, label="Planned EE trajectory")

    # Obstacle as a wireframe sphere
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ox = OBSTACLE_CENTER[0] + OBSTACLE_RADIUS * np.cos(u) * np.sin(v)
    oy = OBSTACLE_CENTER[1] + OBSTACLE_RADIUS * np.sin(u) * np.sin(v)
    oz = OBSTACLE_CENTER[2] + OBSTACLE_RADIUS * np.cos(v)
    ax.plot_wireframe(ox, oy, oz, color="gray", alpha=0.35, linewidth=0.5,
                       label="Obstacle safety zone")

    # Arm pose snapshots at start, 1/3, 2/3, end of the trajectory
    snapshot_idx = sorted(set([0, len(joint_trajectory) // 3,
                                2 * len(joint_trajectory) // 3,
                                len(joint_trajectory) - 1]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_idx)))
    for idx, color in zip(snapshot_idx, colors):
        link_pts = _arm_link_points(joint_trajectory[idx])
        segments = [[link_pts[i], link_pts[i + 1]] for i in range(len(link_pts) - 1)]
        ax.add_collection3d(Line3DCollection(segments, colors=[color], linewidths=3))
        ax.scatter(*link_pts.T, color=color, s=25)

    ax.scatter(*POINT_A, color="lime", s=90, marker="o", label="Point A (start)")
    ax.scatter(*POINT_B, color="red", s=90, marker="X", label="Point B (goal)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    title = "Robotic Arm Trajectory" + (" (Replanned around obstacle)" if replanned else "")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 0.7])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to: {out_path}")


def main():
    print("=" * 70)
    print("PROJECT 1 -- Robotic Arm Kinematics & Path Planning")
    print("=" * 70)
    print(f"Mission: move end-effector from A={POINT_A.tolist()} "
          f"to B={POINT_B.tolist()}")
    print(f"Obstacle: center={OBSTACLE_CENTER.tolist()}, "
          f"radius={OBSTACLE_RADIUS} m\n")

    # --- Process Stage: Cartesian path planning + collision check -----
    waypoints, replanned, plan_log = plan_cartesian_trajectory(
        POINT_A, POINT_B, OBSTACLE_CENTER, OBSTACLE_RADIUS, n_steps=N_WAYPOINTS
    )
    for line in plan_log:
        print(line)

    # --- Process Stage: per-waypoint Inverse Kinematics ----------------
    print(f"\nSolving IK across {len(waypoints)} waypoints...")
    joint_trajectory, all_converged, ik_log = solve_ik_along_trajectory(waypoints)
    for line in ik_log:
        print(line)

    # --- Output Stage: report + FollowJointTrajectory-style summary ----
    status = "SUCCESS" if all_converged else "PARTIAL FAILURE"
    print(f"\n{'🎉' if all_converged else '⚠️ '} {status}: "
          f"trajectory {'fully' if all_converged else 'NOT fully'} "
          f"resolved to joint space.")
    print(f"Path replanned around obstacle: {replanned}")

    final_pos_error = np.linalg.norm(
        POINT_B - forward_kinematics(joint_trajectory[-1])[:3, 3]
    )
    print(f"Final end-effector position error vs Point B: "
          f"{final_pos_error * 1000:.3f} mm")

    print("\nSample joint-angle waypoints (degrees):")
    for idx in [0, len(joint_trajectory) // 2, -1]:
        deg = np.degrees(joint_trajectory[idx])
        print(f"  step {idx:>3}: " + ", ".join(f"J{j+1}={a:7.2f}°" for j, a in enumerate(deg)))

    plot_result(waypoints, joint_trajectory, replanned, "output/arm_trajectory_result.png")


if __name__ == "__main__":
    main()
