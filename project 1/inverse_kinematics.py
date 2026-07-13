"""
inverse_kinematics.py
----------------------
Numerical Inverse Kinematics (IK) for the 6-DOF arm defined in
forward_kinematics.py.

This matches the "Inverse Kinematics (Non-Linear)" + "Local Minima Trap"
+ "Solver Diagnostic Matrix" slides: a single target XYZ can have
multiple valid joint solutions (or none, if unreachable), so we can't
solve it with a straight line of algebra like FK — we iterate.

Solver design (a simplified stand-in for the slide's TRAC-IK):
    - Damped Least Squares (Levenberg-Marquardt style) on the POSITION
      Jacobian. This is the "Orocos-KDL"-style numerical approach from
      the slides, but with a damping term (lambda) added specifically so
      it does NOT fall into the "det(J(q)) -> 0" singularity trap the
      slide deck calls out — that's the whole reason plain
      pseudo-inverse Jacobian solvers only hit ~37% success in the deck's
      diagnostic table.
    - Joint-limit clamping after every iteration ("Joint Limit Clamping"
      slide) so the returned solution is always physically realizable.
    - Warm-starting from the previous waypoint's solution (done by the
      caller, trajectory_planner.py) keeps consecutive IK solves close
      together in joint space, avoiding wild jumps between frames.
"""

import numpy as np

from forward_kinematics import (
    N_JOINTS,
    JOINT_LIMITS,
    forward_kinematics,
    end_effector_position,
)

POSITION_TOLERANCE_M = 1e-3
MAX_ITERATIONS = 200
DAMPING_LAMBDA = 0.02
STEP_SIZE = 1.0
FD_EPSILON = 1e-6


def _numerical_position_jacobian(joint_angles):
    """3x6 Jacobian d(position)/d(joint_angle) via central finite differences."""
    J = np.zeros((3, N_JOINTS))
    for i in range(N_JOINTS):
        dq = np.zeros(N_JOINTS)
        dq[i] = FD_EPSILON
        p_plus = end_effector_position(joint_angles + dq)
        p_minus = end_effector_position(joint_angles - dq)
        J[:, i] = (p_plus - p_minus) / (2 * FD_EPSILON)
    return J


def solve_ik(target_xyz, seed_angles=None, verbose=False):
    """
    Solve for joint angles that place the end-effector at target_xyz.

    Returns:
        (joint_angles, converged: bool, iterations: int, final_error_m: float)
    """
    target_xyz = np.asarray(target_xyz, dtype=float)
    q = np.zeros(N_JOINTS) if seed_angles is None else np.array(seed_angles, dtype=float)

    for iteration in range(1, MAX_ITERATIONS + 1):
        current_pos = end_effector_position(q)
        error = target_xyz - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm < POSITION_TOLERANCE_M:
            if verbose:
                print(f"  IK converged in {iteration - 1} iterations, "
                      f"error={error_norm * 1000:.3f} mm")
            return q, True, iteration - 1, error_norm

        J = _numerical_position_jacobian(q)
        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error
        JJt = J @ J.T
        damped_inv = np.linalg.inv(JJt + (DAMPING_LAMBDA ** 2) * np.eye(3))
        dq = J.T @ damped_inv @ error

        q = q + STEP_SIZE * dq
        # Joint Limit Clamping (as called out on the "Local Minima Trap" slide)
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    final_error = np.linalg.norm(target_xyz - end_effector_position(q))
    if verbose:
        print(f"  IK FAILED to converge in {MAX_ITERATIONS} iterations, "
              f"residual error={final_error * 1000:.3f} mm")
    return q, False, MAX_ITERATIONS, final_error


if __name__ == "__main__":
    # Sanity check: solve IK for a reachable target and verify with FK.
    target = np.array([0.30, -0.15, 0.35])
    q_solution, ok, iters, err = solve_ik(target, verbose=True)
    achieved = end_effector_position(q_solution)
    print("Target:  ", np.round(target, 4))
    print("Achieved:", np.round(achieved, 4))
    print("Converged:", ok, "| Joint angles (deg):", np.round(np.degrees(q_solution), 2))
