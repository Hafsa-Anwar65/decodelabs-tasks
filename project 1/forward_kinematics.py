"""
forward_kinematics.py
----------------------
Forward Kinematics (FK) for a simulated 6-axis (6-DOF revolute) robotic
arm, using the standard Denavit-Hartenberg (DH) convention.

This matches the "Forward Kinematics (Deterministic)" slide: given six
joint angles, FK always yields exactly ONE unique end-effector pose,
solved linearly by chaining DH transformation matrices.

DH table (a, alpha, d are fixed link geometry; theta is the joint
variable, offset by theta_offset so "all zeros" is a sane home pose):

    i | a (m) | alpha (rad) | d (m)  | theta_offset (rad)
    1 | 0.00  | +pi/2       | 0.30   | 0
    2 | 0.25  | 0.00        | 0.00   | -pi/2
    3 | 0.25  | 0.00        | 0.00   | 0
    4 | 0.00  | +pi/2       | 0.10   | -pi/2
    5 | 0.00  | -pi/2       | 0.10   | 0
    6 | 0.00  | 0.00        | 0.10   | 0

(Loosely modeled on a small industrial/training-kit 6-axis arm: a
rotating base, shoulder + elbow "upper/forearm" links, then a 3-DOF
wrist. Units are meters throughout.)
"""

import numpy as np

# a_i, alpha_i, d_i, theta_offset_i
DH_PARAMS = [
    (0.00, np.pi / 2, 0.30, 0.0),
    (0.25, 0.00, 0.00, -np.pi / 2),
    (0.25, 0.00, 0.00, 0.0),
    (0.00, np.pi / 2, 0.10, -np.pi / 2),
    (0.00, -np.pi / 2, 0.10, 0.0),
    (0.00, 0.00, 0.10, 0.0),
]

N_JOINTS = len(DH_PARAMS)

# Symmetric joint limits, degrees -> radians (generic industrial-arm-ish limits)
JOINT_LIMITS_DEG = [
    (-170, 170),
    (-120, 120),
    (-150, 150),
    (-180, 180),
    (-125, 125),
    (-360, 360),
]
JOINT_LIMITS = np.radians(np.array(JOINT_LIMITS_DEG))


def dh_transform(a, alpha, d, theta):
    """Single DH homogeneous transform (i-1)_T_(i)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1],
    ])


def forward_kinematics(joint_angles, return_all_frames=False):
    """
    joint_angles: length-6 array of joint angles (radians), one per axis.

    Returns:
        T (4x4) end-effector pose in the base frame, OR
        (T, frames) if return_all_frames=True, where frames is a list of
        the 4x4 transform of every joint origin (base_link -> joint_i),
        used for plotting the arm's link geometry.
    """
    joint_angles = np.asarray(joint_angles, dtype=float)
    assert joint_angles.shape == (N_JOINTS,), f"expected {N_JOINTS} joint angles"

    T = np.eye(4)
    frames = [T.copy()]
    for (a, alpha, d, theta_offset), q in zip(DH_PARAMS, joint_angles):
        T = T @ dh_transform(a, alpha, d, theta_offset + q)
        frames.append(T.copy())

    if return_all_frames:
        return T, frames
    return T


def end_effector_position(joint_angles):
    """Convenience: just the XYZ position of the end-effector."""
    T = forward_kinematics(joint_angles)
    return T[:3, 3]


if __name__ == "__main__":
    home = np.zeros(N_JOINTS)
    T, frames = forward_kinematics(home, return_all_frames=True)
    print("Home-pose end-effector position (m):", np.round(T[:3, 3], 4))
    print("Home-pose end-effector orientation:\n", np.round(T[:3, :3], 3))
