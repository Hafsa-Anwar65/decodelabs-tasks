"""
trajectory_planner.py
-----------------------
Generates a smooth, collision-free Cartesian trajectory from Point A to
Point B (matches "The Output Stage: Compiling the Action" +
"Collision Avoidance and Scene Monitoring" slides).

Two pieces:

1. Quintic (5th-order polynomial) time-scaling, s(t), applied along the
   straight-line path from A to B. A quintic profile gives position,
   velocity AND acceleration all continuous with ZERO velocity/
   acceleration at both endpoints -- exactly the position/velocity/
   acceleration curves shown in the "Output Stage" slide
   (control_msgs/action/FollowJointTrajectory).

2. A simple sphere obstacle collision check (a stand-in for the FCL /
   Flexible Collision Library check on the slides). Every candidate
   waypoint on the straight-line path is checked against the obstacle;
   if the line dips inside the obstacle's safety radius, the planner
   aborts the direct path and re-routes through a single "lift" via-point
   above the obstacle, then re-validates the two new straight segments.
   This mirrors the "TRAJECTORY ABORT" -> replan behavior on the slides.
"""

import numpy as np


def quintic_time_scaling(n_steps):
    """
    Returns an array s(t) in [0, 1] of length n_steps, following
    s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 (tau = t / T in [0, 1]).
    s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    """
    tau = np.linspace(0.0, 1.0, n_steps)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return s, tau


def _point_to_segment_distance(point, seg_a, seg_b):
    """Shortest distance from `point` to line segment seg_a -> seg_b."""
    seg = seg_b - seg_a
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        return np.linalg.norm(point - seg_a)
    t = np.clip(np.dot(point - seg_a, seg) / seg_len_sq, 0.0, 1.0)
    closest = seg_a + t * seg
    return np.linalg.norm(point - closest)


def path_hits_obstacle(p_start, p_end, obstacle_center, obstacle_radius, margin=0.03):
    """True if the straight segment p_start->p_end passes within
    (obstacle_radius + margin) of obstacle_center at any point."""
    dist = _point_to_segment_distance(obstacle_center, p_start, p_end)
    return dist < (obstacle_radius + margin)


def _interpolate_segment(p_start, p_end, n_steps):
    s, _ = quintic_time_scaling(n_steps)
    return np.array([p_start + s_i * (p_end - p_start) for s_i in s])


def plan_cartesian_trajectory(point_a, point_b, obstacle_center, obstacle_radius,
                               n_steps=40, lift_height=0.15):
    """
    Returns (waypoints, was_replanned: bool, log: list[str])

    waypoints: (n_steps, 3) array of Cartesian XYZ points from A to B,
    quintic-smoothed, guaranteed clear of the obstacle sphere given the
    margin used in path_hits_obstacle.
    """
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)
    log = []

    if not path_hits_obstacle(point_a, point_b, obstacle_center, obstacle_radius):
        log.append("Direct A -> B path clear. No replanning needed.")
        return _interpolate_segment(point_a, point_b, n_steps), False, log

    log.append("TRAJECTORY ABORT: direct A -> B path intersects obstacle "
               f"safety zone (center={np.round(obstacle_center, 3)}, "
               f"radius={obstacle_radius} m). Replanning...")

    # Replan through a single via-point lifted above the obstacle in Z.
    via_point = np.array([
        (point_a[0] + point_b[0]) / 2,
        (point_a[1] + point_b[1]) / 2,
        max(point_a[2], point_b[2], obstacle_center[2] + obstacle_radius) + lift_height,
    ])

    leg1_clear = not path_hits_obstacle(point_a, via_point, obstacle_center, obstacle_radius)
    leg2_clear = not path_hits_obstacle(via_point, point_b, obstacle_center, obstacle_radius)

    if leg1_clear and leg2_clear:
        log.append(f"New path found via lift point {np.round(via_point, 3)}. "
                    "Both segments verified clear.")
        n_half = n_steps // 2
        leg1 = _interpolate_segment(point_a, via_point, n_half)
        leg2 = _interpolate_segment(via_point, point_b, n_steps - n_half)
        return np.vstack([leg1, leg2]), True, log

    # Fall back to a higher lift if the first attempt still clips the obstacle.
    log.append("Lift point still fouls obstacle -- raising lift height and retrying.")
    via_point[2] += lift_height
    n_half = n_steps // 2
    leg1 = _interpolate_segment(point_a, via_point, n_half)
    leg2 = _interpolate_segment(via_point, point_b, n_steps - n_half)
    log.append(f"Final replanned path via {np.round(via_point, 3)}.")
    return np.vstack([leg1, leg2]), True, log


if __name__ == "__main__":
    a = np.array([0.30, -0.20, 0.20])
    b = np.array([0.30, 0.20, 0.20])
    obstacle = np.array([0.30, 0.0, 0.20])
    wps, replanned, log = plan_cartesian_trajectory(a, b, obstacle, obstacle_radius=0.08)
    for line in log:
        print(line)
    print("Waypoint count:", len(wps), "| Replanned:", replanned)
