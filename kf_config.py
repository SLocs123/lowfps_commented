"""Central Kalman filter tuning for low-FPS tracking.

The tracker passes this dictionary into the Kalman filter constructor so all
noise settings live in one place.
"""

KALMAN_KWARGS = {
    # Global multipliers that let you scale whole groups of parameters at once.
    "init_scale": 1.0,
    "process_scale": 1.0,
    "measurement_scale": 1.0,

    # Initial covariance: how uncertain we are when a track is first created.
    "init_pos_xy_weight": 0.18,
    "init_pos_a_std": 0.02,
    "init_pos_h_weight": 0.18,
    "init_vel_xy_weight": 0.12,
    "init_vel_a_std": 2e-5,
    "init_vel_h_weight": 0.12,

    # Process covariance Q: uncertainty added during each prediction step.
    # These values are intentionally higher than usual because 1 FPS leaves
    # more time for an object to move between frames.
    "process_pos_xy_weight": 0.35,
    "process_pos_a_std": 0.07,
    "process_pos_h_weight": 0.22,
    "process_vel_xy_weight": 0.10,
    "process_vel_a_std": 0.0025,
    "process_vel_h_weight": 0.06,

    # Measurement covariance R: how noisy we believe the detector is.
    "measurement_xy_weight": 0.02,
    "measurement_a_std": 0.05,
    "measurement_h_weight": 0.02,
}