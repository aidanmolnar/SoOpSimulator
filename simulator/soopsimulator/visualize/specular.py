"""
Functions for visualizing path of specular point over time.
"""

from simulator.soopsimulator.python_sim_core.specular import RAD_EARTH

import numpy as np
import pyvista as pv


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def get_tube(points, radius=100.0, time=1.0):
    polyline = polyline_from_points(np.nan_to_num(points))
    # polyline = pv.Spline(points, 1000)
    polyline["Time (days)"] = (
        np.arange(polyline.n_points, dtype=float) * time / polyline.n_points
    )
    tube = polyline.tube(radius=radius)
    return tube


def plot_transmitter_receiver_pair(
    plotter,
    receiver_positions,
    transmitter_positions,
    specular_positions,
    time_samples_per_reflect_line=1,
    plot_sphere=True,
    plot_specular_trail=True,
    plot_reflect_lines=True,
    reflect_line_radius=50.0,
    time=1.0,  # Total time of sim to divide revisit count by
    cmap=None,
):
    """
    Uses pyvista to plot one receiver, one transmitter,
    and the path of their specular point over time.
    Can also draw lines representing the reflected signal
    path and a sphere representing the Earth.
    """
    if plot_reflect_lines:
        view_points = np.stack(
            (
                receiver_positions,
                specular_positions,
                specular_positions,
                transmitter_positions,
            ),
            axis=1,
        )
        # Ratio of time samples to reflect lines
        scalars = (
            np.repeat(
                np.arange(view_points.shape[0])[0::time_samples_per_reflect_line], 4
            )
            * time
            / view_points.shape[0]
        )
        view_points = view_points[0::time_samples_per_reflect_line, ...]
        view_points = view_points.reshape(-1, 3)
        reflect_lines = pv.helpers.line_segments_from_points(view_points)
        reflect_lines["Time (days)"] = scalars

    mb = pv.MultiBlock()

    mb.append(get_tube(receiver_positions, time=time))
    mb.append(get_tube(transmitter_positions, time=time))

    if plot_sphere:
        mb.append(pv.Sphere(radius=RAD_EARTH))
    if plot_specular_trail:
        mb.append(get_tube(specular_positions, time=time))
    if plot_reflect_lines:
        mb.append(reflect_lines.tube(radius=reflect_line_radius))

    plotter.add_mesh(mb, smooth_shading=True, opacity=1.0, cmap=cmap)
    plotter.set_background("black")
