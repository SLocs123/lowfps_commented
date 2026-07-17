# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
# i havent changed too much here, just how noise is defined and scaled (large noise increase with each step since low fps)

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state vector is:

        x, y, a, h, vx, vy, va, vh

    where `(x, y)` is the box centre, `a` is aspect ratio, `h` is height, and
    the `v*` terms are the corresponding velocities.

    The model assumes approximately constant velocity between frames.
    Measurements provide only `(x, y, a, h)`, so the velocity terms are hidden
    state that the filter estimates internally.

    """

    def __init__(
        self,
        init_pos_xy_weight=2.0 / 20.0,
        init_pos_a_std=1e-2,
        init_pos_h_weight=2.0 / 20.0,
        init_vel_xy_weight=10.0 / 160.0,
        init_vel_a_std=1e-5,
        init_vel_h_weight=10.0 / 160.0,
        process_pos_xy_weight=1.0 / 20.0,
        process_pos_a_std=5e-2,
        process_pos_h_weight=1.0 / 20.0,
        process_vel_xy_weight=1.0 / 160.0,
        process_vel_a_std=1e-3,
        process_vel_h_weight=1.0 / 160.0,
        measurement_xy_weight=1.0 / 20.0,
        measurement_a_std=1e-1,
        measurement_h_weight=1.0 / 20.0,
        process_scale=1.0,
        measurement_scale=1.0,
        init_scale=1.0,
    ):
        ndim, dt = 4, 1.

        # Build the state transition and observation matrices once.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard-deviation controls for newly created tracks.
        self._init_pos_xy_weight = init_pos_xy_weight
        self._init_pos_a_std = init_pos_a_std
        self._init_pos_h_weight = init_pos_h_weight
        self._init_vel_xy_weight = init_vel_xy_weight
        self._init_vel_a_std = init_vel_a_std
        self._init_vel_h_weight = init_vel_h_weight

        # Standard-deviation controls for the prediction step.
        self._process_pos_xy_weight = process_pos_xy_weight
        self._process_pos_a_std = process_pos_a_std
        self._process_pos_h_weight = process_pos_h_weight
        self._process_vel_xy_weight = process_vel_xy_weight
        self._process_vel_a_std = process_vel_a_std
        self._process_vel_h_weight = process_vel_h_weight

        # Standard-deviation controls for detector measurements.
        self._measurement_xy_weight = measurement_xy_weight
        self._measurement_a_std = measurement_a_std
        self._measurement_h_weight = measurement_h_weight

        # Global multipliers make it easy to scale groups of values together.
        self._process_scale = process_scale
        self._measurement_scale = measurement_scale
        self._init_scale = init_scale

    def initiate(self, measurement, delta = (0, 0)):
        """Create the initial state for a brand-new detection.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            The new mean vector and covariance matrix.

        """
        mean_pos = measurement
        # Start with the observed box and inject an expected initial velocity.
        mean_vel = np.zeros_like(mean_pos)
        mean_vel[0] = delta[0]
        mean_vel[1] = delta[1]
        mean = np.r_[mean_pos, mean_vel]

        std = [
            self._init_pos_xy_weight * measurement[3] * self._init_scale,
            self._init_pos_xy_weight * measurement[3] * self._init_scale,
            self._init_pos_a_std * self._init_scale,
            self._init_pos_h_weight * measurement[3] * self._init_scale,
            self._init_vel_xy_weight * measurement[3] * self._init_scale,
            self._init_vel_xy_weight * measurement[3] * self._init_scale,
            self._init_vel_a_std * self._init_scale,
            self._init_vel_h_weight * measurement[3] * self._init_scale]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Advance the state estimate one time step forward.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            The predicted mean vector and covariance matrix.

        """
        std_pos = [
            self._process_pos_xy_weight * mean[3] * self._process_scale,
            self._process_pos_xy_weight * mean[3] * self._process_scale,
            self._process_pos_a_std * self._process_scale,
            self._process_pos_h_weight * mean[3] * self._process_scale]
        std_vel = [
            self._process_vel_xy_weight * mean[3] * self._process_scale,
            self._process_vel_xy_weight * mean[3] * self._process_scale,
            self._process_vel_a_std * self._process_scale,
            self._process_vel_h_weight * mean[3] * self._process_scale]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Apply the constant-velocity motion model.
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project the hidden state into observable box coordinates.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            The projected mean and covariance in measurement space.

        """
        std = [
            self._measurement_xy_weight * mean[3] * self._measurement_scale,
            self._measurement_xy_weight * mean[3] * self._measurement_scale,
            self._measurement_a_std * self._measurement_scale,
            self._measurement_h_weight * mean[3] * self._measurement_scale]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance, deltas = None):
        """Vectorized prediction for many tracks at once.

        This is faster than predicting each track separately and also allows the
        caller to rescale velocities toward expected lane-direction deltas.

        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            The predicted mean matrix and covariance tensors.
        """
        std_pos = [
            self._process_pos_xy_weight * mean[:, 3] * self._process_scale,
            self._process_pos_xy_weight * mean[:, 3] * self._process_scale,
            self._process_pos_a_std * self._process_scale * np.ones_like(mean[:, 3]),
            self._process_pos_h_weight * mean[:, 3] * self._process_scale]
        std_vel = [
            self._process_vel_xy_weight * mean[:, 3] * self._process_scale,
            self._process_vel_xy_weight * mean[:, 3] * self._process_scale,
            self._process_vel_a_std * self._process_scale * np.ones_like(mean[:, 3]),
            self._process_vel_h_weight * mean[:, 3] * self._process_scale]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        if deltas is not None:
            # Re-scale each velocity so its speed matches the expected lane speed
            # while preserving its current direction.
            target_speed = np.linalg.norm(deltas[:, :2], axis=1)
            velocity = mean[:, 4:6]
            current_speed = np.linalg.norm(velocity, axis=1)

            scale = np.ones_like(target_speed)
            valid = current_speed > 1e-9
            scale[valid] = target_speed[valid] / current_speed[valid]
            velocity *= scale[:, None]
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Correct a prediction using a new detector measurement.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            The corrected mean vector and covariance matrix.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute how far measurements are from the predicted state.

        This is used to reject measurements that are statistically implausible
        given the filter's current uncertainty.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            One distance per measurement.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')