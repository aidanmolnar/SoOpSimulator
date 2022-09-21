from simulator.soopsimulator.constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
)
from simulator.soopsimulator.python_sim_core.specular import find_specular_points
from simulator.soopsimulator.python_sim_core.revisit import (
    setup_count,
    count_on_sphere,
    projection_to_sphere,
    PROJECTION_LENGTH,
)

from specular import plot_transmitter_receiver_pair

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


constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
collections = ConstellationCollection(constellations)

positions = collections.propagate_orbits(PropagationConfig(t_step=0.005, t_range=1.0))

receiver = 30
transmitter = 12

specular = find_specular_points(
    positions[:, receiver : receiver + 1, :],  # noqa
    positions[:, transmitter : transmitter + 1, :],  # noqa
)

count_s, count_n = setup_count(10.0)
count_on_sphere(specular, count_s, count_n)

p = pv.Plotter()

receiver_positions = positions[:, receiver, :]
transmitter_positions = positions[:, transmitter, :]
specular_positions = np.squeeze(specular)
plot_transmitter_receiver_pair(
    p,
    receiver_positions,
    transmitter_positions,
    specular_positions,
    plot_specular_trail=False,
    plot_sphere=False,
    reflect_line_radius=5.0,
)

plot_coverage_3d_hemi(p, count_s, north=False)
plot_coverage_3d_hemi(p, count_n)
p.show()

# TODO: Compare speed of projection as numpy function vs numba function with loop
# TODO: Calculate actual revisit time / medians
# TODO: RUST
