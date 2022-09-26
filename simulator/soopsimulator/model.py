from constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
    OrbitDefinition,
)

from rust_sim_core import find_specular_points, find_revisits
from python_sim_core.angle_constraints import (
    AngleConstraintCalculator,
    AngleConstraintSettings,
)
from python_sim_core.mask_ocean import mask_ocean
from simulator.soopsimulator.constellations import NadirPointing
from visualize.revisit import plot_coverage_3d_hemi, count_at_latitude
from visualize.specular import plot_transmitter_receiver_pair
from visualize.export_vtkjs import export_vtkjs

import pyvista as pv

# TODO: Add revisit statistics
# TODO: Add visualization options
# TODO: Set up chunking?:


class SoOpModel:
    def __init__(
        self,
        receiver_constellations: ConstellationCollection,
        transmitter_constellations: ConstellationCollection,
        propagation_config: PropagationConfig,
    ):
        self.transmitter_constellations = transmitter_constellations
        self.receiver_constellations = receiver_constellations
        self.propagation_config = propagation_config

        self.receiver_positions = None
        self.transmitter_positions = None
        self.specular_positions = None

        self.count_north = None
        self.count_south = None

    def propagate_orbits(self):
        self.transmitter_positions = self.transmitter_constellations.propagate_orbits(
            self.propagation_config
        )
        self.receiver_positions = self.receiver_constellations.propagate_orbits(
            self.propagation_config
        )

    def calculate_specular_points(self):
        if self.receiver_positions is None or self.transmitter_positions is None:
            raise Exception(
                "Must propagate orbits or set receiver/transmitter positions first"
            )

        self.specular_positions = find_specular_points(
            self.receiver_positions,
            self.transmitter_positions,
        )

    def filter_by_angle_constraints(
        self,
        angle_constraint_settings: AngleConstraintSettings,
        max_incidence_angle: float = 90,
    ):
        if self.specular_positions is None:
            raise Exception("Must calculate specular positions before filtering them")

        # Get the antenna cone angles
        (
            tran_nadir_angles,
            tran_zenith_angles,
        ) = self.transmitter_constellations.get_angles()

        (
            recv_nadir_angles,
            recv_zenith_angles,
        ) = self.receiver_constellations.get_angles()

        AngleConstraintCalculator(
            receiver_positions=self.receiver_positions,
            transmitter_positions=self.transmitter_positions,
            specular_positions=self.specular_positions,
            receiver_nadir_cone_angle=recv_nadir_angles,
            receiver_zenith_cone_angle=recv_zenith_angles,
            transmitter_nadir_cone_angle=tran_nadir_angles,
            transmitter_zenith_cone_angle=tran_zenith_angles,
            max_incidence_angle=max_incidence_angle,
            settings=angle_constraint_settings,
        ).filter_specular_points()

    def calculate_visit_counts(self, grid_size=10.0):
        if self.specular_positions is None:
            raise Exception(
                "Must calculate specular positions before calculating revisit counts"
            )

        self.count_south, self.count_north = find_revisits(
            self.specular_positions, grid_size
        )

    def plot_revisit_frequency_3d(self, cmap="magma", clim=None, export=None):
        if self.count_south is None or self.count_north is None:
            raise Exception(
                "Must calculate revisit counts before plotting revisit frequency"
            )

        p = pv.Plotter()
        plot_coverage_3d_hemi(
            p,
            self.count_south / self.propagation_config.t_range,
            north=False,
            cmap=cmap,
            clim=clim,
        )
        plot_coverage_3d_hemi(
            p,
            self.count_north / self.propagation_config.t_range,
            cmap=cmap,
            clim=clim,
        )

        if export is not None:
            export_vtkjs(p, export)

        p.show()

    def plot_transmitter_receiver_pair(
        self,
        receiver=0,
        transmitter=0,
        time_samples_per_reflect_line=1,
        plot_sphere=True,
        plot_specular_trail=True,
        export=None,
        cmap="rainbow",
    ):
        p = pv.Plotter()
        plot_transmitter_receiver_pair(
            p,
            receiver_positions=self.receiver_positions[:, receiver, :],
            transmitter_positions=self.transmitter_positions[:, transmitter, :],
            specular_positions=self.specular_positions[:, receiver, transmitter, :],
            plot_sphere=plot_sphere,
            plot_specular_trail=plot_specular_trail,
            time_samples_per_reflect_line=time_samples_per_reflect_line,
            time=self.propagation_config.t_range,
            cmap=cmap,
        )
        if export is not None:
            export_vtkjs(p, export)

        p.show()

    def mask_oceans(self):
        if self.count_south is None or self.count_north is None:
            raise Exception("Can't mask until counts have been computed")
        self.count_south = self.count_south.astype(float)
        self.count_north = self.count_north.astype(float)
        mask_ocean(self.count_south, self.count_north)

    def plot_latitude_coverage(self, threshold_frequency=1.0):
        threshold_count = threshold_frequency * self.propagation_config.t_range
        count_at_latitude(self.count_north, thresh=threshold_count)

    def plot_pancake_coverage(self):
        # TODO
        pass


if __name__ == "__main__":
    receiver_orbits = []
    for i in range(0, 360, int(360 / 8)):
        orbit = OrbitDefinition(
            e=0.0,
            a=7620.0,
            i=60.0,
            raan=0.0,
            aop=0.0,
            ta=float(i),
        )
        receiver_orbits.append(orbit)

    receiver_constellations = [
        Constellation.from_orbit_definitions(receiver_orbits, NadirPointing(40.0, 40.0))
    ]
    receiver_collections = ConstellationCollection(receiver_constellations)

    transmitter_constellations = [
        Constellation.from_tle(
            "data/TLE/gps.txt", antenna_configuration=NadirPointing(0.0, 90.0)
        ),
        Constellation.from_tle("data/TLE/glonass.txt", NadirPointing(0.0, 90.0)),
        Constellation.from_tle("data/TLE/galileo.txt", NadirPointing(0.0, 90.0)),
        Constellation.from_tle("data/TLE/beidou.txt", NadirPointing(0.0, 90.0)),
        Constellation.from_tle("data/TLE/iridium.txt", NadirPointing(0.0, 90.0)),
    ]
    transmitter_collections = ConstellationCollection(transmitter_constellations)

    model = SoOpModel(
        receiver_collections,
        transmitter_collections,
        PropagationConfig(t_step=1 / (24 * 60), t_range=30.0),
    )

    model.propagate_orbits()
    model.calculate_specular_points()
    model.filter_by_angle_constraints(
        AngleConstraintSettings(
            direct_receiver=True,
            indirect_receiver=True,
            direct_transmitter=True,
            indirect_transmitter=True,
            incidence=True,
        ),
        max_incidence_angle=60.0,
    )
    model.calculate_visit_counts(grid_size=10.0)
    # model.plot_transmitter_receiver_pair()
    # model.mask_oceans()
    model.plot_latitude_coverage(threshold_frequency=0.333)
    model.plot_revisit_frequency_3d(clim=(0, 1 / 1.0))
