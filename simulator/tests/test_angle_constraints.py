from simulator.soopsimulator.python_sim_core.angle_constraints import (
    compute_angle,
    check_antenna_angle,
    AngleConstraintCalculator,
    AngleConstraintsSettings,
)

import pytest
import numpy as np


# Test compute_angle for several single vectors
def test_1():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-4, 2.0, 5.0])
    ang = compute_angle(a, b)
    assert ang == pytest.approx(np.deg2rad(53.30077479))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([-4, 2.0, 5.0])
    ang = compute_angle(a, b)
    assert np.isnan(ang)

    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, -1.0, -1.0])
    ang = compute_angle(a, b)
    assert ang == np.pi


# Test compute_angle for arrays of vectors
def test_2():
    a = np.zeros((2, 1, 3))
    b = np.zeros((1, 2, 3))
    a[0, :, :] = [1.0, 1.0, 1.0]
    a[1, :, :] = [-1.0, -1.0, -1.0]
    b[:, 1, :] = [-1.0, -1.0, -1.0]
    b[:, 0, :] = [1.0, 1.0, 1.0]

    ang = compute_angle(a, b)

    expected_ang = np.array([[0, np.pi], [np.pi, 0]])
    expected_ang = np.expand_dims(expected_ang, axis=-1)

    np.testing.assert_almost_equal(ang, expected_ang)


# Test check_antenna_angle for a couple inputs and nadir/zenith angles
def test_3():
    nadir_cone_angle = np.deg2rad(30)
    zenith_cone_angle = np.deg2rad(20)

    assert check_antenna_angle(np.deg2rad(25), nadir_cone_angle, zenith_cone_angle)
    assert not check_antenna_angle(np.deg2rad(35), nadir_cone_angle, zenith_cone_angle)
    assert not check_antenna_angle(np.deg2rad(155), nadir_cone_angle, zenith_cone_angle)
    assert check_antenna_angle(np.deg2rad(165), nadir_cone_angle, zenith_cone_angle)


# Test check_antenna for arrays of angles and different antenna angles
def test_4():
    angles = np.deg2rad(np.array([20, 90, 160]))
    angles = np.expand_dims(angles, axis=-1)

    nadir_cone_angle = np.deg2rad(np.array([0.0, 30.0]))
    zenith_cone_angle = np.deg2rad(np.array([30.0, 0.0]))

    valid = check_antenna_angle(angles, nadir_cone_angle, zenith_cone_angle)
    expected_valid = np.array(
        [
            [False, True],
            [False, False],
            [True, False],
        ]
    )

    np.testing.assert_equal(valid, expected_valid)


# Tests for direct receiver
def test_5():
    # Transmitter must be inside 45 degree cone oriented radially starting from receiver
    def filter(R, T, S):
        settings = AngleConstraintsSettings(
            direct_receiver=True,
            direct_transmitter=False,
            indirect_receiver=False,
            indirect_transmitter=False,
            incidence=False,
        )

        AngleConstraintCalculator(
            receiver_positions=R,
            transmitter_positions=T,
            specular_positions=S,
            receiver_nadir_cone_angle=0.0,
            receiver_zenith_cone_angle=45.0,
            transmitter_nadir_cone_angle=0.0,
            transmitter_zenith_cone_angle=0.0,
            max_incidence_angle=0.0,
            settings=settings,
        ).filter_specular_points()

    R = np.reshape(np.array([0.0, 1.0, 0.0]), (1, 1, 3))
    T = np.reshape(np.array([1.0, 2.01, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 1.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert not np.isnan(S).any()

    R = np.reshape(np.array([0.0, 1.0, 0.0]), (1, 1, 3))
    T = np.reshape(np.array([1.0, 1.99, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 1.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert np.isnan(S).all()


# Two tests for indirect receiver
def test_6():
    # Specular point must be in 45 degree angle downwards from receiver
    def filter(R, T, S):
        settings = AngleConstraintsSettings(
            direct_receiver=False,
            direct_transmitter=False,
            indirect_receiver=True,
            indirect_transmitter=False,
            incidence=False,
        )

        AngleConstraintCalculator(
            receiver_positions=R,
            transmitter_positions=T,
            specular_positions=S,
            receiver_nadir_cone_angle=45.0,
            receiver_zenith_cone_angle=0.0,
            transmitter_nadir_cone_angle=0.0,
            transmitter_zenith_cone_angle=0.0,
            max_incidence_angle=0.0,
            settings=settings,
        ).filter_specular_points()

    R = np.reshape(np.array([0.0, 0.0, 10.0]), (1, 1, 3))
    T = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([9.0, 0.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert not np.isnan(S).any()

    R = np.reshape(np.array([0.0, 0.0, 10.0]), (1, 1, 3))
    T = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 11.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert np.isnan(S).all()


# Two tests for direct transmitter
def test_7():
    # Receiver must be in a 45 degree cone oriented vertically downward from transmitter
    def filter(R, T, S):
        settings = AngleConstraintsSettings(
            direct_receiver=False,
            direct_transmitter=True,
            indirect_receiver=False,
            indirect_transmitter=False,
            incidence=False,
        )

        AngleConstraintCalculator(
            receiver_positions=R,
            transmitter_positions=T,
            specular_positions=S,
            receiver_nadir_cone_angle=0.0,
            receiver_zenith_cone_angle=0.0,
            transmitter_nadir_cone_angle=45.0,
            transmitter_zenith_cone_angle=0.0,
            max_incidence_angle=0.0,
            settings=settings,
        ).filter_specular_points()

    T = np.reshape(np.array([0.0, 0.0, 10.0]), (1, 1, 3))
    R = np.reshape(np.array([9.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert not np.isnan(S).any()

    T = np.reshape(np.array([0.0, 0.0, 10.0]), (1, 1, 3))
    R = np.reshape(np.array([0.0, 11.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert np.isnan(S).all()


# Two tests for indirect transmitter
def test_8():
    # Specular point must be in 45 degree angle downwards from transmitter
    def filter(R, T, S):
        settings = AngleConstraintsSettings(
            direct_receiver=False,
            direct_transmitter=False,
            indirect_receiver=False,
            indirect_transmitter=True,
            incidence=False,
        )

        AngleConstraintCalculator(
            receiver_positions=R,
            transmitter_positions=T,
            specular_positions=S,
            receiver_nadir_cone_angle=0.0,
            receiver_zenith_cone_angle=0.0,
            transmitter_nadir_cone_angle=45.0,
            transmitter_zenith_cone_angle=0.0,
            max_incidence_angle=0.0,
            settings=settings,
        ).filter_specular_points()

    T = np.reshape(np.array([0.0, 0.0, 10.0]), (1, 1, 3))
    R = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([9.0, 0.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    print(S)
    assert not np.isnan(S).any()

    T = np.reshape(np.array([0.0, 0.0, 10.0]), (1, 1, 3))
    R = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 11.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert np.isnan(S).all()


# Two tests for incidence angle
def test_9():
    # Angle between path from receiver to specular point and
    #   specular radial normal must be less than 45 degrees
    def filter(R, T, S):
        settings = AngleConstraintsSettings(
            direct_receiver=False,
            direct_transmitter=False,
            indirect_receiver=False,
            indirect_transmitter=False,
            incidence=True,
        )

        AngleConstraintCalculator(
            receiver_positions=R,
            transmitter_positions=T,
            specular_positions=S,
            receiver_nadir_cone_angle=0.0,
            receiver_zenith_cone_angle=0.0,
            transmitter_nadir_cone_angle=0.0,
            transmitter_zenith_cone_angle=0.0,
            max_incidence_angle=45.0,
            settings=settings,
        ).filter_specular_points()

    R = np.reshape(np.array([-5.0, 10.0, 0.0]), (1, 1, 3))
    T = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 4.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert not np.isnan(S).any()

    R = np.reshape(np.array([-5.0, 10.0, 0.0]), (1, 1, 3))
    T = np.reshape(np.array([0.0, 0.0, 0.0]), (1, 1, 3))
    S = np.reshape(np.array([0.0, 6.0, 0.0]), (1, 1, 1, 3))
    filter(R, T, S)
    assert np.isnan(S).all()


# Check combinations of angle constraints
def test_10():

    R = np.zeros((4, 1, 3))
    R[0, :, :] = [0.0, 0.0, 5.0]
    R[1, :, :] = [0.0, 0.0, 5.0]
    R[2, :, :] = [0.0, 0.0, 5.0]
    R[3, :, :] = [0.0, 0.0, 5.0]

    T = np.zeros((4, 1, 3))
    T[0, :, :] = [4.9, 0.0, 10.0]  # Both valid
    T[1, :, :] = [5.1, 0.0, 10.0]  # Transmitter invalid
    T[2, :, :] = [4.9, 0.0, 0.0]
    T[3, :, :] = [5.1, 0.0, 10.0]  # Both invalid

    S = np.zeros((4, 1, 1, 3))
    S[0, :, :, :] = [3.9, 0.0, 1.0]  # Both valid
    S[1, :, :, :] = [3.9, 0.0, 1.0]
    S[2, :, :, :] = [4.1, 0.0, 1.0]  # Specular point invalid
    S[3, :, :, :] = [4.1, 0.0, 1.0]  # Both invalid

    # Transmitter and specular point must both be in 45 degree cone above or below
    settings = AngleConstraintsSettings(
        direct_receiver=True,
        direct_transmitter=False,
        indirect_receiver=True,
        indirect_transmitter=False,
        incidence=False,
    )
    AngleConstraintCalculator(
        receiver_positions=R,
        transmitter_positions=T,
        specular_positions=S,
        receiver_nadir_cone_angle=45.0,
        receiver_zenith_cone_angle=45.0,
        transmitter_nadir_cone_angle=0.0,
        transmitter_zenith_cone_angle=0.0,
        max_incidence_angle=0.0,
        settings=settings,
    ).filter_specular_points()

    # Check only when both conditions are met is point valid
    assert not np.isnan(S[0, ...]).any()
    assert np.isnan(S[1:4, ...]).all()


# Integrated test (multiple antenna configurations, receivers, transmitters, times)
def test_11():

    R = np.zeros((4, 2, 3))
    # 45 degree up receiver
    R[0, 0, :] = [0.0, 10.0, 0.0]
    R[1, 0, :] = [0.0, 10.0, 0.0]
    R[2, 0, :] = [0.0, 10.0, 0.0]
    R[3, 0, :] = [0.0, 10.0, 0.0]

    # 45 degree down receiver
    R[0, 1, :] = [0.0, 10.0, 0.0]
    R[1, 1, :] = [0.0, 10.0, 0.0]
    R[2, 1, :] = [0.0, 10.0, 0.0]
    R[3, 1, :] = [0.0, 10.0, 0.0]

    T = np.zeros((4, 1, 3))
    T[0, 0, :] = [0.0, 11.0, 0.0]  # Above receiver
    T[1, 0, :] = [0.0, 9.0, 0.0]  # Below receiver
    T[2, 0, :] = [0.0, 9.0, 0.0]
    T[3, 0, :] = [0.0, 11.0, 0.0]  # Above receiver

    S = np.zeros((4, 2, 1, 3))
    # 45 degree up receiver
    S[0, 0, :, :] = [0.0, 11.0, 0.0]  # Above up receiver
    S[1, 0, :, :] = [0.0, 11.0, 0.0]  # Above up receiver
    S[2, 0, :, :] = [0.0, 9.0, 0.0]  # below up receiver (invalid)
    S[3, 0, :, :] = [0.0, 11.0, 0.0]  # Above up receiver

    # 45 degree down receiver
    S[0, 1, :, :] = [0.0, 9.0, 0.0]  # Below down receiver
    S[1, 1, :, :] = [0.0, 9.0, 0.0]  # Below down receiver
    S[2, 1, :, :] = [0.0, 11.0, 0.0]  # above down receiver (invalid)
    S[3, 1, :, :] = [0.0, 9.0, 0.0]  # below down receiver

    # Test two constraints
    settings = AngleConstraintsSettings(
        direct_receiver=True,
        direct_transmitter=False,
        indirect_receiver=True,
        indirect_transmitter=False,
        incidence=False,
    )

    # Two receivers:
    #   one with a 45 degree cone pointing up
    #   one with a 45 degree cone pointing down
    AngleConstraintCalculator(
        receiver_positions=R,
        transmitter_positions=T,
        specular_positions=S,
        receiver_nadir_cone_angle=np.array([0.0, 45.0]),
        receiver_zenith_cone_angle=np.array([45.0, 0.0]),
        transmitter_nadir_cone_angle=0.0,
        transmitter_zenith_cone_angle=0.0,
        max_incidence_angle=0.0,
        settings=settings,
    ).filter_specular_points()

    # fmt: off
    assert not np.isnan(S[0, 0, 0, 0])
    assert     np.isnan(S[1, 0, 0, 0]) # noqa # transmitter below
    assert     np.isnan(S[2, 0, 0, 0]) # noqa # specular below
    assert not np.isnan(S[3, 0, 0, 0])

    assert     np.isnan(S[0, 1, 0, 0]) # noqa # transmitter above
    assert not np.isnan(S[1, 1, 0, 0])
    assert     np.isnan(S[2, 1, 0, 0]) # noqa # specular above
    assert     np.isnan(S[3, 1, 0, 0]) # noqa # transmitter above
    # fmt: on
