import numpy as np
from global_land_mask import globe
from .revisit import PROJECTION_LENGTH, RAD_EARTH, projection_to_sphere


def mask_ocean(count_south: np.ndarray, count_north: np.ndarray):
    pixels_per_side = count_south.shape[0]
    edges = np.linspace(
        -0.5 * PROJECTION_LENGTH, 0.5 * PROJECTION_LENGTH, pixels_per_side + 1
    )
    mean_a = np.convolve(edges, [0.5, 0.5], mode="valid")
    mean_b = np.expand_dims(mean_a, axis=0)
    mean_a = np.expand_dims(mean_a, axis=-1)

    x, y, z = projection_to_sphere(mean_a, mean_b, True)

    lats = np.rad2deg(np.arccos(-z / RAD_EARTH) - np.pi / 2)
    long = np.rad2deg(np.arctan2(y, x))

    ocean_south = globe.is_ocean(-lats, long)
    ocean_north = globe.is_ocean(lats, long)

    count_south[ocean_south] = np.nan
    count_north[ocean_north] = np.nan
