from simulator.soopsimulator.constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
)
from simulator.soopsimulator.python_sim_core.specular import RAD_EARTH

# from rust_sim_core import find_specular_points
from simulator.soopsimulator.python_sim_core.specular_numba import find_specular_points

from .export_vtkjs import export_vtkjs


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
    time=1.0,
    cmap=None,
):
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


def plot_gps_pair():
    constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
    collections = ConstellationCollection(constellations)

    points = collections.propagate_orbits(PropagationConfig(t_step=0.005, t_range=1.0))

    receiver = 0
    transmitter = 1

    receiver_positions = points[:, receiver, :]
    transmitter_positions = points[:, transmitter, :]

    specular = find_specular_points(
        points[:, receiver : receiver + 1, :],  # noqa
        points[:, transmitter : transmitter + 1, :],  # noqa
    )
    specular_positions = np.squeeze(specular)

    pv.global_theme.cmap = "jet"
    p = pv.Plotter()

    plot_transmitter_receiver_pair(
        p,
        receiver_positions,
        transmitter_positions,
        specular_positions,
    )

    export_vtkjs(p, "demosite/src/scenes/gps")
    p.show()

    # Adds a slider that updates lines showing path of reflected signal
    def add_slider():
        def create_line(index):
            index = int(index)

            R = points[index, receiver, :]
            T = points[index, transmitter, :]
            S = specular[index, :]

            p.add_mesh(
                pv.Line(R, S).tube(radius=100.0),
                smooth_shading=True,
                name="line1",
                color="w",
            )
            p.add_mesh(
                pv.Line(S, T).tube(radius=100.0),
                smooth_shading=True,
                name="line2",
                color="w",
            )

        p.add_slider_widget(create_line, [0, points.shape[0] - 1], title="")


def plot_gps_with_iridium():
    prop_config = PropagationConfig(t_step=0.001, t_range=0.5)

    receiver_constellations = [Constellation.from_tle("data/TLE/gps.txt", None)]
    receiver_collections = ConstellationCollection(receiver_constellations)
    transmitter_positions = receiver_collections.propagate_orbits(prop_config)

    receiver_constellations = [Constellation.from_tle("data/TLE/iridium.txt", None)]
    receiver_collections = ConstellationCollection(receiver_constellations)
    receiver_positions = receiver_collections.propagate_orbits(prop_config)

    receiver = 0
    transmitter = 0

    specular = find_specular_points(
        transmitter_positions,  # noqa
        receiver_positions,  # noqa
    )
    specular_positions = np.squeeze(specular)  # noqa

    pv.global_theme.cmap = "jet"
    p = pv.Plotter()

    plot_transmitter_receiver_pair(
        p,
        receiver_positions[:, receiver, :],
        transmitter_positions[:, transmitter, :],
        specular_positions[:, receiver, transmitter, :],
        # time_samples_per_reflect_line=5,
    )

    export_vtkjs(p, "demosite/src/scenes/gps_and_iridium")
    p.show()


if __name__ == "__main__":
    plot_gps_pair()
    plot_gps_with_iridium()
