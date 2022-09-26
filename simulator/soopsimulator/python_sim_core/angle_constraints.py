import numpy as np
from dataclasses import dataclass


# What angle constraints to evaluate
@dataclass
class AngleConstraintSettings:
    direct_receiver: bool = True
    direct_transmitter: bool = True
    indirect_receiver: bool = True
    indirect_transmitter: bool = True
    incidence: bool = True


class AngleConstraintCalculator:
    # receiver_posiitions should be array with shape (times, receivers, 3)
    # transmitter_posiitions should be array with shape (times, transmitters, 3)
    # specular_posiitions should be array with shape (times, receivers, transmitters, 3)
    def __init__(
        self,
        receiver_positions: np.ndarray,
        transmitter_positions: np.ndarray,
        specular_positions: np.ndarray,
        receiver_nadir_cone_angle: np.ndarray,
        receiver_zenith_cone_angle: np.ndarray,
        transmitter_nadir_cone_angle: np.ndarray,
        transmitter_zenith_cone_angle: np.ndarray,
        max_incidence_angle: float,
        settings: AngleConstraintSettings,
    ):
        # Take positions as references
        self.R = receiver_positions[...]
        self.T = transmitter_positions[...]
        self.S = specular_positions[...]

        self.validate_position_dimensions()

        # Convert angles to degrees
        self.max_incidence_angle = np.deg2rad(max_incidence_angle)
        self.receiver_nadir_cone_angle = np.deg2rad(receiver_nadir_cone_angle)
        self.receiver_zenith_cone_angle = np.deg2rad(receiver_zenith_cone_angle)
        self.transmitter_nadir_cone_angle = np.deg2rad(transmitter_nadir_cone_angle)
        self.transmitter_zenith_cone_angle = np.deg2rad(transmitter_zenith_cone_angle)

        self.validate_cone_angle_dimensions()

        self.R = np.expand_dims(self.R, axis=2)
        self.T = np.expand_dims(self.T, axis=1)
        # R is array of receiver    positions with shape (time, num recv, 1,         3)
        # T is array of transmitter positions with shape (time, 1,        num trans, 3)
        # S is array of receiver    positions with shape (time, num recv, num trans, 3)

        # Reshape cone angles
        self.receiver_nadir_cone_angle = self.receiver_nadir_cone_angle.reshape(
            (1, -1, 1, 1)
        )
        self.receiver_zenith_cone_angle = self.receiver_zenith_cone_angle.reshape(
            (1, -1, 1, 1)
        )
        self.transmitter_nadir_cone_angle = self.transmitter_nadir_cone_angle.reshape(
            (1, 1, -1, 1)
        )
        self.transmitter_zenith_cone_angle = self.transmitter_zenith_cone_angle.reshape(
            (1, 1, -1, 1)
        )

        self.settings = settings

    def validate_position_dimensions(self):
        assert self.R.shape[0] == self.T.shape[0]
        assert self.T.shape[0] == self.S.shape[0]

        assert self.R.shape[1] == self.S.shape[1]
        assert self.T.shape[2] == self.T.shape[2]

        assert self.R.shape[-1] == 3
        assert self.T.shape[-1] == 3
        assert self.S.shape[-1] == 3

        assert len(self.R.shape) == 3
        assert len(self.T.shape) == 3
        assert len(self.S.shape) == 4

    # Cone angles should be floats or 1 dimensional
    #   arrays with a value for each transmitter / receiver
    def validate_cone_angle_dimensions(self):
        assert len(self.receiver_nadir_cone_angle.shape) == 0 or (
            len(self.receiver_nadir_cone_angle.shape) == 1
            and self.receiver_nadir_cone_angle.shape[0] == self.R.shape[1]
        )
        assert len(self.receiver_zenith_cone_angle.shape) == 0 or (
            len(self.receiver_zenith_cone_angle.shape) == 1
            and self.receiver_zenith_cone_angle.shape[0] == self.R.shape[1]
        )
        assert len(self.transmitter_nadir_cone_angle.shape) == 0 or (
            len(self.transmitter_nadir_cone_angle.shape) == 1
            and self.transmitter_nadir_cone_angle.shape[0] == self.T.shape[1]
        )
        assert len(self.transmitter_zenith_cone_angle.shape) == 0 or (
            len(self.transmitter_zenith_cone_angle.shape) == 1
            and self.transmitter_zenith_cone_angle.shape[0] == self.T.shape[1]
        )

    def filter_specular_points(self):
        valid = np.ones(self.S.shape, dtype=bool)

        # For all computed angles:
        #  0 is nadir, pi is zenith

        if self.settings.direct_receiver:
            valid = np.bitwise_and(
                valid,
                check_antenna_angle(
                    compute_angle(-(self.T - self.R), self.R),
                    self.receiver_nadir_cone_angle,
                    self.receiver_zenith_cone_angle,
                ),
            )

        if self.settings.direct_transmitter:
            valid = np.bitwise_and(
                valid,
                check_antenna_angle(
                    compute_angle(self.T - self.R, self.T),
                    self.transmitter_nadir_cone_angle,
                    self.transmitter_zenith_cone_angle,
                ),
            )

        if self.settings.indirect_receiver:
            valid = np.bitwise_and(
                valid,
                check_antenna_angle(
                    compute_angle(self.R - self.S, self.R),
                    self.receiver_nadir_cone_angle,
                    self.receiver_zenith_cone_angle,
                ),
            )

        if self.settings.indirect_transmitter:
            valid = np.bitwise_and(
                valid,
                check_antenna_angle(
                    compute_angle(self.T - self.S, self.T),
                    self.transmitter_nadir_cone_angle,
                    self.transmitter_zenith_cone_angle,
                ),
            )

        if self.settings.incidence:
            angle = compute_angle(self.R - self.S, self.S)
            valid = np.bitwise_and(valid, angle < self.max_incidence_angle)

        self.S[~valid] = np.nan


# Checks if an angle falls below the nadir angle or above the zenith angle
def check_antenna_angle(angle, nadir_angle, zenith_angle):
    return (angle < nadir_angle) | (angle > (np.pi - zenith_angle))


# Calculates angle between two satellite positions (in last dimension) using dot product
# Range is 0 to pi
def compute_angle(a, b):
    mag_a = np.linalg.norm(a, axis=-1)
    mag_b = np.linalg.norm(b, axis=-1)
    cos = np.einsum("...i,...i->...", a, b) / (mag_a * mag_b)
    cos = np.clip(cos, -1, 1)  # Clip within arccos range to avoid floating point nans
    cos = np.expand_dims(cos, axis=-1)
    return np.arccos(cos)
