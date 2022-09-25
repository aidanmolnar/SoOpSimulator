from simulator.soopsimulator.constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
)
from simulator.soopsimulator.python_sim_core.specular import (
    # find_specular_points,
    RAD_EARTH,
)

from rust_sim_core import find_specular_points

from export_vtkjs import export_vtkjs

import numpy as np
import pyvista as pv


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def get_tube(points, radius=100.0):
    polyline = polyline_from_points(points)
    polyline["scalars"] = np.arange(polyline.n_points)
    tube = polyline.tube(radius=radius)
    # tube.plot(smooth_shading=True)
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
        scalars = np.repeat(
            np.arange(view_points.shape[0])[0::time_samples_per_reflect_line], 4
        )
        view_points = view_points[0::time_samples_per_reflect_line, ...]
        view_points = view_points.reshape(-1, 3)
        reflect_lines = pv.helpers.line_segments_from_points(view_points)
        reflect_lines["scalars"] = scalars

    mb = pv.MultiBlock()

    mb.append(get_tube(receiver_positions))
    mb.append(get_tube(transmitter_positions))

    if plot_sphere:
        mb.append(pv.Sphere(radius=RAD_EARTH))
    if plot_specular_trail:
        mb.append(get_tube(specular_positions))
    if plot_reflect_lines:
        mb.append(reflect_lines.tube(radius=reflect_line_radius))

    plotter.add_mesh(mb, smooth_shading=True)
    plotter.set_background("black")


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


if __name__ == "__main__":
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

    print(specular_positions)

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
