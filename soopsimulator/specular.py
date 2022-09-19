import numpy as np
from numpy.lib.scimath import sqrt


RAD_EARTH = 6371  # Approximate radius of earth (km)


# Compute the specular reflection points between all pairs of z1 and z2
# Assumes a perfectly spherical earth
# Based on this: https://www.geometrictools.com/Documentation/SphereReflections.pdf
# z1 shape: (t, n_z1, 3)
# z2 shape: (t, n_z2, 3)
def find_specular_points(z1, z2, rad_earth=RAD_EARTH):
    # fmt: off

    # Store array shapes
    n_t  = z1.shape[0] # noqa
    n_z1 = z1.shape[1]
    n_z2 = z2.shape[1]

    # Normalize points by radius of earth
    z1 = z1 / rad_earth
    z2 = z2 / rad_earth

    # Reshape arrays for broadcasting
    z1 = np.expand_dims(z1, 2)
    z2 = np.expand_dims(z2, 1)

    # Compute the dot products in the reference
    a = np.reshape(np.einsum("ijkl,ijkl->ij",  z1, z1), (n_t, n_z1,    1)) # noqa
    b = np.reshape(np.einsum("ijkl,ijkl->ijk", z1, z2), (n_t, n_z1, n_z2))
    c = np.reshape(np.einsum("ijkl,ijkl->ik",  z2, z2), (n_t, 1,    n_z2))

    # Coefficients of quartic equation from reference
    A = 4. * c * (a * c - b**2.)
    B = -4. * (a * c - b**2.)
    C = a + 2. * b + c - 4. * a * c
    D = 2. * (a - b)
    E = a - 1.

    y = solve_quartic(A, B, C, D, E)

    # Reshape arrays for broadcasting
    y  = np.expand_dims(y,  axis=-2) # noqa
    z1 = np.expand_dims(z1, axis=-1)
    z2 = np.expand_dims(z2, axis=-1)
    c = np.reshape(c, (n_t, 1,    n_z2, 1, 1))
    b = np.reshape(b, (n_t, n_z1, n_z2, 1, 1))

    x = (-2 * c * y**2 + y + 1) / (2 * b * y + 1)

    # Find potential specular points
    N = x * z1 + y * z2
    N = N / np.linalg.norm(N, axis=-2, keepdims=True)

    # So at this point N is a an array of (t, n_S, n_L, xyz, roots)
    # There are 4 possible solutions to the quartic equation along the roots
    # axis and we need to pick the valid one (shortest light path).

    # Total length of light path
    q = np.linalg.norm(z1 - N, axis=-2, keepdims=True) + \
        np.linalg.norm(N - z2, axis=-2, keepdims=True)

    # Choose root that minimizes length of light path.
    least_q = np.argmin(q, axis=-1, keepdims=True)
    N = np.take_along_axis(N, least_q, axis=-1)

    # Remove roots dimension
    N = np.squeeze(N, axis=-1)
    z1 = np.squeeze(z1, axis=-1)
    z2 = np.squeeze(z2, axis=-1)

    # It's possible that some of the specular points found are on the inside
    #  of the sphere.
    # Remove them by checking the dot product of N and both z1 and z2 is positive
    dot_z1 = np.reshape(np.einsum("ijkl,ijkl->ijk", N, z1), (n_t, n_z1, n_z2))
    dot_z2 = np.reshape(np.einsum("ijkl,ijkl->ijk", N, z2), (n_t, n_z1, n_z2))

    invalid = np.logical_or(dot_z1 < 0, dot_z2 < 0)
    invalid = np.stack((invalid, invalid, invalid), axis=-1)
    N[invalid] = np.NaN

    # If z1 or z2 < rad_earth, invalid specular points may be present in output

    N = N * rad_earth
    return N

    # fmt: on


# Gets roots of quartic equation analytically for arrays of coefficients.
# ! Returns only real part of roots !
# ! Does not handle degenerate cases (cubics, quadratics, etc) !
# ! Does not handle 4x repeated roots !
# These cases do not occur often enough to impact validitiy and fail
# convservatively (assume point was invalid).
# Would hurt performance to account for them.
# Uses Ferrari's solution (https://en.wikipedia.org/wiki/Quartic_equation).
# Could probably be optimized further.
def solve_quartic(A, B, C, D, E):
    # Initialize roots.
    # Add extra column for each root of quartic
    A = np.array(A)
    roots = np.zeros(A.shape + (4,), dtype=float)

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

    U = R.astype(complex) ** (1.0 / 3.0)

    y = np.zeros(U.shape, dtype=complex)

    y1 = -(5.0 / 6.0) * alpha - np.cbrt(Q)

    # This can return divide by 0 errors, but if U is zero the other
    # y is chosen.
    with np.errstate(divide="ignore"):
        y2 = -(5.0 / 6.0) * alpha + U - P / (3 * U)

    y[U == 0.0] = y1[U == 0.0]
    y[U != 0.0] = y2[U != 0.0]

    W = sqrt(alpha + 2.0 * y)

    r0 = -B / (4.0 * A) + (-W - sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / W))) / 2.0
    r1 = -B / (4.0 * A) + (+W - sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / W))) / 2.0
    r2 = -B / (4.0 * A) + (-W + sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / W))) / 2.0
    r3 = -B / (4.0 * A) + (+W + sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / W))) / 2.0

    roots[..., 0] = r0.real
    roots[..., 1] = r1.real
    roots[..., 2] = r2.real
    roots[..., 3] = r3.real

    return roots
