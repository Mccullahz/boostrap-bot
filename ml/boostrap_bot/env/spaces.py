"""
observation and action space definitions.

these dimensions MUST stay in sync with the Go inference layer:
- Go will allocate buffers of size OBS_SIZE and ACTION_SIZE.
- observation layout: [ball_xyz, ball_vel_xyz, car_xyz, car_vel_xyz, car_rot_xyz, car_ang_vel_xyz, car_boost, ...]
- action layout: [throttle, steer, pitch, yaw, roll, jump, boost]
"""

import numpy as np

# ---- Observation space (single vector for the controlled car + game state) ----
# Ball: position (3), velocity (3)
# Car: position (3), velocity (3), rotation euler (3), angular velocity (3), boost (1)
# Optional: pad or add opponent / game time for future use
BALL_FEATURES = 6   # x,y,z, vx,vy,vz
CAR_FEATURES = 13   # x,y,z, vx,vy,vz, pitch,yaw,roll, ax,ay,az, boost
OBS_PAD = 6         # reserved for game time, ball touch, etc.
OBS_SIZE = BALL_FEATURES + CAR_FEATURES + OBS_PAD  # 25

# ---- Action space (RLBot ControllerState-aligned) ----
# throttle, steer, pitch, yaw, roll in [-1, 1]; jump, boost in [0, 1]
ACTION_SIZE = 7


def obs_space_shape():
    return (OBS_SIZE,)


def action_space_shape():
    return (ACTION_SIZE,)


def obs_space_bounds():
    """Approximate bounds for normalization (field ~±4096, vel ~±2300, angles ±pi)."""
    low = np.array(
        [-4096.0] * 3 + [-2300.0] * 3 +   # ball pos, vel
        [-4096.0] * 3 + [-2300.0] * 3 + [-np.pi] * 3 + [-5.5] * 3 + [0.0] +  # car
        [-1.0] * OBS_PAD,
        dtype=np.float32,
    )
    high = np.array(
        [4096.0] * 3 + [2300.0] * 3 +
        [4096.0] * 3 + [2300.0] * 3 + [np.pi] * 3 + [5.5] * 3 + [100.0] +
        [1.0] * OBS_PAD,
        dtype=np.float32,
    )
    return low, high


def action_space_bounds():
    low = np.array([-1.0] * 5 + [0.0, 0.0], dtype=np.float32)
    high = np.array([1.0] * 5 + [1.0, 1.0], dtype=np.float32)
    return low, high
