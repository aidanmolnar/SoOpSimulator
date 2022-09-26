"""
Signals of opportunity simulator.  An abstraction over the other code here.
"""

from constellations import (
    ConstellationCollection,
    PropagationConfig,
)

from rust_sim_core import find_specular_points, find_revisits
from python_sim_core.angle_constraints import (
    AngleConstraintCalculator,
    AngleConstraintSettings,
)
from python_sim_core.mask_ocean import mask_ocean
from visualize.revisit import plot_coverage_3d_hemi, count_at_latitude
from visualize.specular import plot_transmitter_receiver_pair
from visualize.export_vtkjs import export_vtkjs

import pyvista as pv


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
        """
        Calculate the positions of the transmitters and receivers over the given time
        range and step size.
        """
        self.transmitter_positions = self.transmitter_constellations.propagate_orbits(
            self.propagation_config
        )
        self.receiver_positions = self.receiver_constellations.propagate_orbits(
            self.propagation_config
        )

    def calculate_specular_points(self):
        """
        Finds the specular reflection point between every combination of transmitter
        and receiver at every time step.
        Drops specular points that have an incidence angle of more than 90 degrees
        (intersects with sphere).
        """
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
        """
        Removes specular points that don't meet a set of angle constraints
        (sets them to NaN).
        """
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
        """
        Counts the number of times a specular path passes through every cell over the
        entire earth.
        grid_size is the aproximate side length of the cells in km.
        Uses a line drawing algorithm to ensure all cells intersect by the
        path are incremented.
        """
        if self.specular_positions is None:
            raise Exception(
                "Must calculate specular positions before calculating revisit counts"
            )

        self.count_south, self.count_north = find_revisits(
            self.specular_positions, grid_size
        )

    def plot_revisit_frequency_3d(self, cmap="magma", clim=None, export=None):
        """
        Generates a 3d plot of the revisit frequency using pyvista.
        """
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
        """
        Plots a single transmitter and receiver and their specular path over all time
        steps. Draws lines representing the path of the reflected signal between the
        transmitter and receiver.  Positions are in the IRTS Earth-Centered Earth-Fixed
        reference frame.
        """
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
        """
        Sets the revisit counts of any cells over the ocean to NaN.
        """
        if self.count_south is None or self.count_north is None:
            raise Exception("Can't mask until counts have been computed")
        self.count_south = self.count_south.astype(float)
        self.count_north = self.count_north.astype(float)
        mask_ocean(self.count_south, self.count_north)

    def plot_latitude_coverage(self, threshold_frequency=1.0):
        """
        Plots the fraction of cells that meet a revisit frequency threshold vs latitude.
        """
        threshold_count = threshold_frequency * self.propagation_config.t_range
        count_at_latitude(self.count_north, thresh=threshold_count)

    def plot_pancake_coverage(self):
        # TODO Similar to plot_revisit_frequency_3d,
        # but the hemispheres would flattened to circles for easier viewing.
        pass
