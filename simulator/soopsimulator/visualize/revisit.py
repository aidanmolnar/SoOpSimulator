from simulator.soopsimulator.constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
)

from simulator.soopsimulator.python_sim_core.revisit import (
    count_on_sphere,
    projection_to_sphere,
    PROJECTION_LENGTH,
    RAD_EARTH,
    setup_count,
)

from rust_sim_core import find_revisits, find_specular_points

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def plot_coverage_3d_hemi(
    plotter,
    data,
    north=True,
    data_name="Revisit Frequency (visit/day)",
    cmap=None,
    clim=None,
):
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
    grid[data_name] = data.transpose().reshape(-1)
    grid.dimensions = (squares, squares, 1)
    plotter.add_mesh(
        grid,
        style="surface",
        smooth_shading=True,
        cmap=cmap,
        nan_color="blue",
        clim=clim,
    )
    plotter.set_background("black")


def count_at_latitude(count, thresh=0, greater_than=True):
    d = count.shape[0] - 1

    counts = np.zeros(int(d / 2) + 1)
    squares = np.ones(int(d / 2) + 1)

    for i in range(0, int(d / 2) + 1 - ((d + 1) % 2)):
        # fmt: off
        if greater_than:
            counts[i] = np.sum(count[i:d-i+1, i      ] > thresh) + \
                        np.sum(count[i:d-i+1, d-i    ] > thresh) + \
                        np.sum(count[i,       i+1:d-i] > thresh) + \
                        np.sum(count[d-i,     i+1:d-i] > thresh) # noqa 
        else:
            counts[i] = np.sum(count[i:d-i+1, i      ] < thresh) + \
                        np.sum(count[i:d-i+1, d-i    ] < thresh) + \
                        np.sum(count[i,       i+1:d-i] < thresh) + \
                        np.sum(count[d-i,     i+1:d-i] < thresh) # noqa 
        squares[i] = 4*(d-2*i)
        # fmt: on

    if (d + 1) % 2:
        counts[int(d / 2)] = count[int(d / 2), int(d / 2)]
        squares[i] = 1

    edges = np.linspace(-0.5 * PROJECTION_LENGTH, 0, int((d + 1) / 2) + 1 + (d + 1) % 2)
    mean_a = np.convolve(edges, [0.5, 0.5], mode="valid")

    _, _, z = projection_to_sphere(mean_a, 1e-8, True)

    lats = np.rad2deg(np.arccos(-z / RAD_EARTH) - np.pi / 2)

    plt.plot(lats, counts / squares)
    plt.title("Coverage vs Latitude")
    plt.xlabel("Latitude")
    plt.ylabel("Fraction of Cells that Meet Requirement")
    plt.show()


def plot_gps_pair():
    from specular import plot_transmitter_receiver_pair

    constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
    collections = ConstellationCollection(constellations)

    positions = collections.propagate_orbits(
        PropagationConfig(t_step=0.005, t_range=1.0)
    )

    receiver = 30
    transmitter = 12

    speculars = find_specular_points(
        positions[:, receiver : receiver + 1, :],  # noqa
        positions[:, transmitter : transmitter + 1, :],  # noqa
    )

    # count_s, count_n = setup_count(10.0)
    count_s, count_n = find_revisits(speculars, 10.0)

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

    receiver_constellations = [Constellation.from_tle("data/TLE/iridium.txt", None)]
    receiver_collections = ConstellationCollection(receiver_constellations)
    receiver_positions = receiver_collections.propagate_orbits(prop_config)

    transmitter_constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
    transmitter_collections = ConstellationCollection(transmitter_constellations)
    transmitter_positions = transmitter_collections.propagate_orbits(prop_config)

    speculars = find_specular_points(
        transmitter_positions,  # noqa
        receiver_positions,  # noqa
    )

    import time

    grid_size = 10.0

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
    plot_gps_pair()
