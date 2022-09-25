from simulator.soopsimulator.constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
)

from simulator.soopsimulator.python_sim_core.revisit import (
    count_on_sphere,
    projection_to_sphere,
    PROJECTION_LENGTH,
    setup_count,
)

from specular import plot_transmitter_receiver_pair
from rust_sim_core import find_revisits, find_specular_points

import numpy as np
import pyvista as pv


def plot_coverage_3d_hemi(
    plotter, data, north=True, points=None, show=True, bgcolor=(0, 0, 0)
):
    # data = data.astype(np.float32)

    squares = data.shape[0]

    edges = np.linspace(-0.5 * PROJECTION_LENGTH, 0.5 * PROJECTION_LENGTH, squares)
    edges[edges == 0] = 1e-8

    # Get the xyz coordinates of a grid of points projected to sphere
    points_2d = np.reshape(np.meshgrid(edges, edges), (2, -1)).T
    a = points_2d[:, 0]
    b = points_2d[:, 1]
    north = north * np.ones(len(a))
    x, y, z = projection_to_sphere(a, b, north)

    # surf = mlab.pipeline.surface(n_mesh, vmin=0, vmax=vmax)
    grid = pv.StructuredGrid(x, y, z)
    grid["num_visits"] = data.transpose().reshape(-1)
    grid.dimensions = (squares, squares, 1)
    plotter.add_mesh(grid, style="surface", smooth_shading=True)


def plot_gps_pair():
    constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
    collections = ConstellationCollection(constellations)

    positions = collections.propagate_orbits(
        PropagationConfig(t_step=0.001, t_range=1.0)
    )

    receiver = 30
    transmitter = 12

    speculars = find_specular_points(
        positions[:, receiver : receiver + 1, :],  # noqa
        positions[:, transmitter : transmitter + 1, :],  # noqa
    )

    # count_s, count_n = setup_count(10.0)
    count_s, count_n = find_revisits(speculars)

    p = pv.Plotter()

    receiver_positions = positions[:, receiver, :]
    transmitter_positions = positions[:, transmitter, :]
    specular_positions = np.squeeze(speculars)
    plot_transmitter_receiver_pair(
        p,
        receiver_positions,
        transmitter_positions,
        specular_positions,
        plot_specular_trail=False,
        plot_sphere=False,
        reflect_line_radius=25.0,
    )

    plot_coverage_3d_hemi(p, count_s, north=False)
    plot_coverage_3d_hemi(p, count_n)
    p.show()


def plot_iridium_gps():
    prop_config = PropagationConfig(t_step=0.001, t_range=1.0)

    receiver_constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
    receiver_collections = ConstellationCollection(receiver_constellations)
    transmitter_positions = receiver_collections.propagate_orbits(prop_config)

    receiver_constellations = [Constellation.from_tle("data/TLE/iridium.txt", None)]
    receiver_collections = ConstellationCollection(receiver_constellations)
    receiver_positions = receiver_collections.propagate_orbits(prop_config)

    speculars = find_specular_points(
        transmitter_positions,  # noqa
        receiver_positions,  # noqa
    )

    import time

    grid_size = 1.0

    start = time.time()
    count_s, count_n = setup_count(grid_size)
    count_on_sphere(speculars, count_s, count_n)
    print("Numba: %.2f (s)" % (time.time() - start))

    start = time.time()
    count_s, count_n = find_revisits(speculars, grid_size)
    print("Rust: %.2f (s)" % (time.time() - start))

    p = pv.Plotter()

    plot_coverage_3d_hemi(p, count_s, north=False)
    plot_coverage_3d_hemi(p, count_n)
    p.show()


# TODO: Compare speed of projection as numpy function vs numba function with loop
# TODO: Calculate actual revisit time / medians
# TODO: RUST
if __name__ == "__main__":
    plot_iridium_gps()
