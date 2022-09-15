from soopsimulator.specular import solve_quartic, find_specular_points, RAD_EARTH
import numpy as np


# Test solving several quartics
# Coefificents from expansion in wolfram alpha
def test_1():
    # Small distinct coefficients
    roots = solve_quartic(1.0, -2.0, -13.0, 14.0, 24.0)
    roots = np.sort(roots, axis=-1)
    np.testing.assert_array_almost_equal(roots, [-3.0, -1.0, 2.0, 4.0])

    # leading coefficient greater than 1
    roots = solve_quartic(2.0, -4.0, -26.0, 28.0, 48.0)
    roots = np.sort(roots, axis=-1)
    np.testing.assert_array_almost_equal(roots, [-3.0, -1.0, 2.0, 4.0])

    # 2x repeated root
    roots = solve_quartic(1.0, 7.0, -12.0, -68.0, 112)
    roots = np.sort(roots, axis=-1)
    np.testing.assert_array_almost_equal(roots, [-7.0, -4.0, 2.0, 2.0])

    # 3x repeated root
    roots = solve_quartic(1.0, -14.0, 36.0, -34.0, 11.0)
    roots = np.sort(roots, axis=-1)
    np.testing.assert_array_almost_equal(roots, [1.0, 1.0, 1.0, 11.0])

    # Large and fractional values
    roots = solve_quartic(1.0, -73135.1, -3.67215e8, 3.9642e11, 4.32441e13)
    roots = np.sort(roots, axis=-1)
    np.testing.assert_array_almost_equal(
        roots,
        [-5555.55, -100.007, 1000.5655, 77790.1],
        decimal=2,  # Some loss of accuracy
    )


# Test solving quartics with array of inputs
def test_2():
    A = np.array([1.0, 1.0])
    B = np.array([-2.0, -4.2])
    C = np.array([-13.0, -74.2])
    D = np.array([14.0, 90.6])
    E = np.array([24.0, 428.4])
    roots = solve_quartic(A, B, C, D, E)
    roots = np.sort(roots, axis=-1)
    np.testing.assert_array_almost_equal(roots[0], [-3.0, -1.0, 2.0, 4.0])
    np.testing.assert_array_almost_equal(roots[1], [-7.0, -2.0, 3.0, 10.2])


# Test finding specular point
def test_3():
    n_t = 1
    n_R = 1
    n_T = 1
    R = np.zeros((n_t, n_R, 3))
    T = np.zeros((n_t, n_T, 3))

    R[0, 0, 0] = 10000
    T[0, 0, 1] = 10000
    S = find_specular_points(R, T)

    # R and T are at 90 degree angle and equal distance away.
    # Result should be between on surface of Earth
    S_expect = RAD_EARTH * np.array([np.cos(np.deg2rad(45)), np.cos(np.deg2rad(45)), 0])
    np.testing.assert_array_almost_equal(S[0, 0, 0, :], S_expect)


# Test finding specular points (90 degrees apart in different octants)
# Multiple times
def test3():
    n_t = 3
    n_R = 1
    n_T = 1
    R = np.zeros((n_t, n_R, 3))
    T = np.zeros((n_t, n_T, 3))

    R[0, 0, 0] = 10000
    T[0, 0, 1] = 10000
    R[1, 0, 0] = -10000
    T[1, 0, 1] = 10000
    R[2, 0, 0] = 10000
    T[2, 0, 2] = -10000
    S = find_specular_points(R, T)

    # R and T are at 90 degree angle and equal distance away.
    # Result should be between on surface of Earth
    S_expect = RAD_EARTH * np.array([np.cos(np.deg2rad(45)), np.cos(np.deg2rad(45)), 0])
    np.testing.assert_array_almost_equal(S[0, 0, 0, :], S_expect)

    S_expect = RAD_EARTH * np.array(
        [-np.cos(np.deg2rad(45)), np.cos(np.deg2rad(45)), 0]
    )
    np.testing.assert_array_almost_equal(S[1, 0, 0, :], S_expect)

    S_expect = RAD_EARTH * np.array(
        [np.cos(np.deg2rad(45)), 0.0, -np.cos(np.deg2rad(45))]
    )
    np.testing.assert_array_almost_equal(S[2, 0, 0, :], S_expect)


# Test more complicated specular arangement
def test4():
    n_t = 1
    n_R = 1
    n_T = 1
    R = np.zeros((n_t, n_R, 3))
    T = np.zeros((n_t, n_T, 3))

    R[0, 0, :] = [1300.0, 18000.0, 0.0]
    T[0, 0, :] = [9000.0, 2100.0, 0.0]
    S = find_specular_points(R, T)

    # Used 2d Solidworks CAD sketch to get S_expect
    S_expect = np.array([4923.35, 4054.72, 0.0])
    np.testing.assert_array_almost_equal(S[0, 0, 0, :], S_expect, decimal=2)


# Test more complicated specular arangement with large values.
def test5():
    n_t = 1
    n_R = 1
    n_T = 1
    R = np.zeros((n_t, n_R, 3))
    T = np.zeros((n_t, n_T, 3))

    R[0, 0, :] = [1300.0, 18000.0, 0.0]
    T[0, 0, :] = [90000.0, 2100.0, 0.0]

    # Used 2d Solidworks CAD sketch to get S_expect
    S = find_specular_points(R, T)

    S_expect = np.array([4092.69, 4891.83, 0.0])
    np.testing.assert_array_almost_equal(S[0, 0, 0, :], S_expect, decimal=2)


# Test arrangement that has invalid specular point
def test6():
    n_t = 1
    n_R = 1
    n_T = 1
    R = np.zeros((n_t, n_R, 3))
    T = np.zeros((n_t, n_T, 3))

    R[0, 0, :] = [10000.0, 100.0, 0.0]
    T[0, 0, :] = [-10000.0, 100.0, 0.0]

    # Used 2d Solidworks CAD sketch to get S_expect
    S = find_specular_points(R, T)

    S_expect = np.array([np.nan, np.nan, np.nan])
    np.testing.assert_array_almost_equal(S[0, 0, 0, :], S_expect, decimal=2)


# Test multiple transmitters and receivers
def test7():
    n_t = 1
    n_R = 2
    n_T = 2
    R = np.zeros((n_t, n_R, 3))
    T = np.zeros((n_t, n_T, 3))

    R[0, 0, :] = [10000.0, 0.0, 0.0]
    R[0, 1, :] = [-10000.0, 0.0, 0.0]

    T[0, 0, :] = [0.0, -10000.0, 0.0]
    T[0, 1, :] = [0.0, 0.0, 10000.0]
    S = find_specular_points(R, T)

    S_expect = RAD_EARTH * np.array(
        [np.cos(np.deg2rad(45)), -np.cos(np.deg2rad(45)), 0],
    )
    np.testing.assert_array_almost_equal(S[0, 0, 0, :], S_expect)

    S_expect = RAD_EARTH * np.array(
        [-np.cos(np.deg2rad(45)), -np.cos(np.deg2rad(45)), 0],
    )
    np.testing.assert_array_almost_equal(S[0, 1, 0, :], S_expect)

    S_expect = RAD_EARTH * np.array(
        [np.cos(np.deg2rad(45)), 0.0, np.cos(np.deg2rad(45))],
    )
    np.testing.assert_array_almost_equal(S[0, 0, 1, :], S_expect)

    S_expect = RAD_EARTH * np.array(
        [-np.cos(np.deg2rad(45)), 0.0, np.cos(np.deg2rad(45))],
    )
    np.testing.assert_array_almost_equal(S[0, 1, 1, :], S_expect)
