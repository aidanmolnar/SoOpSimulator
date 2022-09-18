import numpy as np

import pyvista as pv

from constellations import Constellation, ConstellationCollection, PropagationConfig
from specular import find_specular_points, RAD_EARTH
from export_vtkjs import export_vtkjs

constellations = [Constellation.from_tle("TLE/gps.txt", None)]
collections = ConstellationCollection(constellations)

points = collections.propagate_orbits(PropagationConfig(t_step=0.005, t_range=1.0))

receiver = 0
transmitter = 1

specular = find_specular_points(
    points[:, receiver : receiver + 1, :],  # noqa
    points[:, transmitter : transmitter + 1, :],  # noqa
)
specular = np.squeeze(specular)
print(specular.shape)


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


view_points = np.stack(
    (points[:, receiver, :], specular, specular, points[:, transmitter, :]), axis=1
)
skip = 2
scalars = np.repeat(np.arange(view_points.shape[0])[0::skip], 4)
view_points = view_points[0::skip, ...]
view_points = view_points.reshape(-1, 3)

reflect_lines = pv.helpers.line_segments_from_points(view_points)
reflect_lines["scalars"] = scalars


pv.global_theme.cmap = "rainbow"
p = pv.Plotter()


data = {
    "sphere": pv.Sphere(radius=RAD_EARTH),
    "receiver": get_tube(points[:, receiver, :]),
    "transmitter": get_tube(points[:, transmitter, :]),
    "specular": get_tube(specular),
    "reflect": reflect_lines.tube(radius=50.0),
}

mb = pv.MultiBlock(data)

p.add_mesh(mb, smooth_shading=True)
p.set_background("black")

export_vtkjs(p, "site/src/scenes/gps")
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
