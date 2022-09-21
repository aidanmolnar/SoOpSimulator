import numpy as np
from numba import njit


ROOT_PI = np.sqrt(np.pi)
PI = np.pi

RAD_EARTH = 6371
PROJECTION_LENGTH = RAD_EARTH * np.sqrt(2 * np.pi)


# https://www.aanda.org/articles/aa/pdf/2010/12/aa15278-10.pdf
@njit(inline="always", cache=True)
def sphere_to_projection(x, y, z):
    # x = S[..., 0]
    # y = S[..., 1]
    # z = S[..., 2]

    north = z > 0
    boo = np.abs(y) <= np.abs(x)

    mult = north * np.sqrt(2 * RAD_EARTH * (RAD_EARTH - z)) + ~north * np.sqrt(
        2 * RAD_EARTH * (RAD_EARTH + z)
    )

    a = mult * (
        boo * np.sign(x) * ROOT_PI / 2
        + ~boo * np.sign(y) * (2 / ROOT_PI) * np.arctan(x / y)
    )
    b = mult * (
        boo * np.sign(x) * (2 / ROOT_PI) * np.arctan(y / x)
        + ~boo * np.sign(y) * ROOT_PI / 2
    )

    return a, b


# https://www.aanda.org/articles/aa/pdf/2010/12/aa15278-10.pdf
@njit(inline="always", cache=True)
def projection_to_sphere(a, b, north):
    boo = np.abs(b) <= np.abs(a)

    x = boo * (2 * a / PI) * np.sqrt(PI - a**2 / RAD_EARTH**2) * np.cos(
        b * PI / (4 * a)
    ) + ~boo * (2 * b / PI) * np.sqrt(PI - b**2 / RAD_EARTH**2) * np.sin(
        a * PI / (4 * b)
    )
    y = boo * (2 * a / PI) * np.sqrt(PI - a**2 / RAD_EARTH**2) * np.sin(
        b * PI / (4 * a)
    ) + ~boo * (2 * b / PI) * np.sqrt(PI - b**2 / RAD_EARTH**2) * np.cos(
        a * PI / (4 * b)
    )
    z = boo * ((2 * a**2) / (PI * RAD_EARTH) - RAD_EARTH) + ~boo * (
        (2 * b**2) / (PI * RAD_EARTH) - RAD_EARTH
    )

    z *= 1 - 2 * north

    return x, y, z


# Implemented based on:
# https://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
@njit(inline="always", cache=True)
def plotline(count, x0, y0, x1, y1):
    # Rescale coordinates to grid used for line algorithm
    x0 = (x0 / PROJECTION_LENGTH + 0.5) * count.shape[0]
    y0 = (y0 / PROJECTION_LENGTH + 0.5) * count.shape[0]
    x1 = (x1 / PROJECTION_LENGTH + 0.5) * count.shape[0]
    y1 = (y1 / PROJECTION_LENGTH + 0.5) * count.shape[0]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    x = int(np.floor(x0))
    y = int(np.floor(y0))

    n = 1

    if dx == 0:
        x_inc = 0
        error = np.inf
    elif x1 > x0:
        x_inc = 1
        n += int(np.floor(x1)) - x
        error = (np.floor(x0) + 1 - x0) * dy
    else:
        x_inc = -1
        n += x - int(np.floor(x1))
        error = (x0 - np.floor(x0)) * dy

    if dy == 0:
        y_inc = 0
        error = -np.inf
    elif y1 > y0:
        y_inc = 1
        n += int(np.floor(y1)) - y
        error -= (np.floor(y0) + 1 - y0) * dx
    else:
        y_inc = -1
        n += y - int(np.floor(y1))
        error -= (y0 - np.floor(y0)) * dx

    # Skips end point so interpolated segments don't double count
    while n > 1:
        if 0 <= x < count.shape[0] and 0 <= y < count.shape[0]:
            # Plot the pixel
            count[x, y] += 1

        if error > 0:
            y += y_inc
            error -= dx
        else:
            x += x_inc
            error += dy

        n -= 1


@njit(cache=True)
def count_on_sphere(S, count_s, count_n):
    # Iterate over every combination of transmitter and receiver
    for j in range(0, S.shape[1]):
        for k in range(0, S.shape[2]):

            # Iterate over time
            for i in range(1, S.shape[0]):
                x0 = S[i - 1, j, k, 0]
                y0 = S[i - 1, j, k, 1]
                z0 = S[i - 1, j, k, 2]
                x1 = S[i, j, k, 0]
                y1 = S[i, j, k, 1]
                z1 = S[i, j, k, 2]

                # Don't render the line if one of the coordinates is not valid
                if np.isnan(x0) or np.isnan(x1):
                    continue

                # Find the fraction of the way through the line when it
                #   crosses hemispheres
                if z0 == z1:
                    t = 0.5
                else:
                    # From linear interpolation;
                    # z = z0 + t*(z1-z0) = 0
                    t = z0 / (z0 - z1)

                # Projection coordinates
                a0, b0 = sphere_to_projection(x0, y0, z0)
                a1, b1 = sphere_to_projection(x1, y1, z1)

                # Check if line crosses hemisphers
                if 0 < t < 1:
                    # Interpolated point where line crosses hemispheres
                    xm = x0 + (x1 - x0) * t
                    ym = y0 + (y1 - y0) * t
                    zm = z0 + (z1 - z0) * t

                    # Projection of cross point
                    am, bm = sphere_to_projection(xm, ym, zm)

                    if z0 < 0:
                        plotline(count_s, a0, b0, am, bm)
                        plotline(count_n, am, bm, a1, b1)
                    else:
                        plotline(count_n, a0, b0, am, bm)
                        plotline(count_s, am, bm, a1, b1)
                else:
                    # Line is exclusively in one hemisphere
                    if z0 < 0:
                        plotline(count_s, a0, b0, a1, b1)
                    else:
                        plotline(count_n, a0, b0, a1, b1)


def setup_count(grid_size: float) -> tuple[np.ndarray, np.ndarray]:
    num_divisions = int(PROJECTION_LENGTH / grid_size)

    count_s = np.zeros((num_divisions, num_divisions), dtype=np.uint16)
    count_n = np.zeros((num_divisions, num_divisions), dtype=np.uint16)

    return (count_s, count_n)
