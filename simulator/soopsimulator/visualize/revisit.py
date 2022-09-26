"""
Functions for visualizing revisit frequency.
"""

from simulator.soopsimulator.python_sim_core.revisit import (
    projection_to_sphere,
    PROJECTION_LENGTH,
    RAD_EARTH,
)

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def plot_coverage_3d_hemi(
    plotter,
    data,  # see sim_core.reviist for details of data format
    north=True,
    data_name="Revisit Frequency (visit/day)",
    cmap=None,
    clim=None,
):
    """
    Plots the revisit frequency over a hemisphere using pyvista.
    """
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
    """
    Counts the number of cells at a given latitude that have more revisits
    than a given threshold.  Plots the fraction of cells vs latitude that
    meet the threshold.
    """

    d = count.shape[0] - 1

    counts = np.zeros(int(d / 2) + 1)
    squares = np.ones(int(d / 2) + 1)

    # In the projection, bands of latitude are perimeters of a square.
    # 0 latitude is the outer perimeter of the data.
    # This sums along each of the side lengths of the square as the square
    # gets smaller.
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

    # Correction for the middle square if the side length is even
    if (d + 1) % 2:
        counts[int(d / 2)] = count[int(d / 2), int(d / 2)]
        squares[i] = 1

    # Computes the mean x coordinate of each cell
    edges = np.linspace(-0.5 * PROJECTION_LENGTH, 0, int((d + 1) / 2) + 1 + (d + 1) % 2)
    mean_a = np.convolve(edges, [0.5, 0.5], mode="valid")

    # Finds the latitude of the cell from the x coordinate of the projection
    _, _, z = projection_to_sphere(mean_a, 1e-8, True)
    lats = np.rad2deg(np.arccos(-z / RAD_EARTH) - np.pi / 2)

    # Plots revisit coverage vs latitude
    plt.plot(lats, counts / squares)
    plt.title("Coverage vs Latitude")
    plt.xlabel("Latitude")
    plt.ylabel("Fraction of Cells that Meet Requirement")
    plt.show()
