import numpy as np
from cmath import sqrt
from numba import njit, prange


RAD_EARTH = 6371  # Approximate radius of earth (km)


# Simple dot product between two 3d vectors
@njit(inline="always")
def dot(a_x, a_y, a_z, b_x, b_y, b_z):
    return a_x * b_x + a_y * b_y + a_z * b_z


@njit(inline="always")
def norm(a_x, a_y, a_z):
    return np.sqrt(dot(a_x, a_y, a_z, a_x, a_y, a_z))


# Compute the specular reflection points between all pairs of z1 and z2
# Assumes a perfectly spherical earth
# Based on this: https://www.geometrictools.com/Documentation/SphereReflections.pdf
# z1 shape: (t, n_z1, 3)
# z2 shape: (t, n_z2, 3)
@njit(cache=True)
def find_specular_points(z1, z2):
    n_t = z1.shape[0]  # noqa
    n_z1 = z1.shape[1]
    n_z2 = z2.shape[1]

    speculars = np.empty((n_t, n_z1, n_z2, 3))

    for i in prange(0, n_t):
        for j in range(0, n_z1):
            for k in range(0, n_z2):
                recv_x = z1[i, j, 0] / RAD_EARTH
                recv_y = z1[i, j, 1] / RAD_EARTH
                recv_z = z1[i, j, 2] / RAD_EARTH
                trans_x = z2[i, k, 0] / RAD_EARTH
                trans_y = z2[i, k, 1] / RAD_EARTH
                trans_z = z2[i, k, 2] / RAD_EARTH

                a = dot(recv_x, recv_y, recv_z, recv_x, recv_y, recv_z)
                b = dot(recv_x, recv_y, recv_z, trans_x, trans_y, trans_z)
                c = dot(trans_x, trans_y, trans_z, trans_x, trans_y, trans_z)

                # Coefficients of quartic equation from reference
                A = 4.0 * c * (a * c - b**2.0)
                B = -4.0 * (a * c - b**2.0)
                C = a + 2.0 * b + c - 4.0 * a * c
                D = 2.0 * (a - b)
                E = a - 1.0

                roots = solve_quartic(A, B, C, D, E)

                q_min = np.inf
                spec_final_x = np.nan
                spec_final_y = np.nan
                spec_final_z = np.nan

                for y in roots:

                    x = (-2 * c * y**2 + y + 1) / (2 * b * y + 1)

                    # Find potential specular points
                    spec_x = x * recv_x + y * trans_x
                    spec_y = x * recv_y + y * trans_y
                    spec_z = x * recv_z + y * trans_z

                    # Normalize specular points
                    norm_spec = norm(spec_x, spec_y, spec_z)
                    spec_x = spec_x / norm_spec
                    spec_y = spec_y / norm_spec
                    spec_z = spec_z / norm_spec

                    r_s_x = recv_x - spec_x
                    r_s_y = recv_y - spec_y
                    r_s_z = recv_z - spec_z
                    s_t_x = spec_x - trans_x
                    s_t_y = spec_y - trans_y
                    s_t_z = spec_z - trans_z

                    # Total length of light path
                    q = norm(r_s_x, r_s_y, r_s_z) + norm(s_t_x, s_t_y, s_t_z)

                    if q < q_min:
                        q_min = q
                        spec_final_x = spec_x
                        spec_final_y = spec_y
                        spec_final_z = spec_z

                if (
                    dot(
                        recv_x - spec_final_x,
                        recv_y - spec_final_y,
                        recv_z - spec_final_z,
                        spec_final_x,
                        spec_final_y,
                        spec_final_z,
                    )
                    > 0.0
                    and dot(
                        trans_x - spec_final_x,
                        trans_y - spec_final_y,
                        trans_z - spec_final_z,
                        spec_final_x,
                        spec_final_y,
                        spec_final_z,
                    )
                    > 0.0
                ):
                    speculars[i, j, k, 0] = spec_final_x * RAD_EARTH
                    speculars[i, j, k, 1] = spec_final_y * RAD_EARTH
                    speculars[i, j, k, 2] = spec_final_z * RAD_EARTH
                else:
                    speculars[i, j, k, 0] = np.nan
                    speculars[i, j, k, 1] = np.nan
                    speculars[i, j, k, 2] = np.nan

    return speculars


# Gets roots of quartic equation analytically for arrays of coefficients.
# ! Returns only real part of roots !
# ! Does not handle degenerate cases (cubics, quadratics, etc) !
# ! Does not handle 4x repeated roots !
# These cases do not occur often enough to impact validitiy and fail
# convservatively (assume point was invalid).
# Would hurt performance to account for them.
# Uses Ferrari's solution (https://en.wikipedia.org/wiki/Quartic_equation).
# Could probably be optimized further.
@njit(inline="always")
def solve_quartic(A, B, C, D, E):
    roots = np.zeros(4, dtype=float)

    alpha = (-3.0 / 8.0) * B**2.0 / A**2.0 + C / A
    beta = (1.0 / 8.0) * B**3.0 / A**3.0 - B * C / (2.0 * A**2.0) + D / A
    gam = (
        (-3.0 / 256.0) * B**4.0 / A**4.0
        + C * B**2.0 / (16.0 * A**3)
        - B * D / (4.0 * A**2.0)
        + E / A
    )

    P = -(1.0 / 12.0) * alpha**2.0 - gam
    Q = -(1.0 / 108.0) * alpha**3.0 + alpha * gam / 3.0 - beta**2.0 / 8.0

    R = -Q / 2.0 + sqrt(Q**2.0 / 4.0 + P**3.0 / 27.0)

    U = R ** (1.0 / 3.0)

    if U == 0.0:
        y = -(5.0 / 6.0) * alpha - np.cbrt(Q)
    else:
        y = -(5.0 / 6.0) * alpha + U - P / (3 * U)

    W = sqrt(alpha + 2.0 * y)

    r0 = -B / (4.0 * A) + (-W - sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / W))) / 2.0
    r1 = -B / (4.0 * A) + (+W - sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / W))) / 2.0
    r2 = -B / (4.0 * A) + (-W + sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / W))) / 2.0
    r3 = -B / (4.0 * A) + (+W + sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / W))) / 2.0

    roots[0] = r0.real
    roots[1] = r1.real
    roots[2] = r2.real
    roots[3] = r3.real

    return roots
