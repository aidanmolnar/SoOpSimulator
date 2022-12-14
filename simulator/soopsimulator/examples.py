"""
Examples demonstrating use of simulator and visualization options.
"""

from constellations import (
    Constellation,
    OrbitDefinition,
    NadirPointing,
)

from model import (
    ConstellationCollection,
    SoOpModel,
    PropagationConfig,
    AngleConstraintSettings,
)

# TODO:
# One gps pair scene
# One gps-iridium scene
# One skynet coverage scene


def gps_pair_scene():
    """
    Plots the specular trail between two GPS satellites.
    """
    receiver_constellations = [
        Constellation.from_tle(
            "data/TLE/gps.txt",
            antenna_configuration=None,
            indices_to_use=[5],
        ),
    ]
    receiver_collections = ConstellationCollection(receiver_constellations)

    transmitter_constellations = [
        Constellation.from_tle(
            "data/TLE/gps.txt",
            antenna_configuration=None,
            indices_to_use=[0],
        ),
    ]
    transmitter_collections = ConstellationCollection(transmitter_constellations)

    model = SoOpModel(
        receiver_collections,
        transmitter_collections,
        PropagationConfig(t_step=0.005, t_range=1.0),
    )

    model.propagate_orbits()
    model.calculate_specular_points()
    model.plot_transmitter_receiver_pair(export="demosite/src/scenes/gps_pair")


def galileo_with_iridium_scene():
    """
    Plots the specular trail between a galileo gnss satelite and an iridium satelite.
    """
    receiver_constellations = [
        Constellation.from_tle(
            "data/TLE/galileo.txt",
            antenna_configuration=None,
            indices_to_use=[1],
        ),
    ]
    receiver_collections = ConstellationCollection(receiver_constellations)

    transmitter_constellations = [
        Constellation.from_tle(
            "data/TLE/iridium.txt",
            antenna_configuration=None,
            indices_to_use=[10],
        ),
    ]
    transmitter_collections = ConstellationCollection(transmitter_constellations)

    model = SoOpModel(
        receiver_collections,
        transmitter_collections,
        PropagationConfig(t_step=0.0025, t_range=0.25),
    )

    model.propagate_orbits()
    model.calculate_specular_points()
    model.plot_transmitter_receiver_pair(
        export="demosite/src/scenes/galileo_with_iridium"
    )


def bulk_coverage_scene():
    """
    Calculates the revisit frequency of a constellation of 8 receiver satellites
    evenly spread by true anomaly with an orbit inclination of 60 degrees
    and a semi major axis of 7620 km.  Various L-band transmitter constellations
    are used (orbits loaded from TLES from Celestrak). Receivers have a 40 degree
    antenna cone angle in the nadir and zenith direction to receive direct/indirect
    signals.  If the direct and indirect signals do not enter one of these cones
    the specular point is discarded.
    """
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
    model.plot_latitude_coverage(threshold_frequency=0.333)
    model.mask_oceans()
    model.plot_revisit_frequency_3d(clim=(0, 1 / 1.0))


def skynet_coverage_scene():
    """
    Calculates the revisit frequency of a constellation of 8 receiver satellites
    evenly spread by true anomaly with an orbit inclination of 60 degrees
    and a semi major axis of 7620 km.  5 skynet transmitters in GEO are used.
    Revisit coverage is limited to where the transmitters have visibility.
    """
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
            "data/TLE/skynet.txt", antenna_configuration=NadirPointing(0.0, 90.0)
        ),
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
    model.plot_latitude_coverage(threshold_frequency=0.333)
    model.mask_oceans()
    model.plot_revisit_frequency_3d(clim=(0, 1 / 1.0), cmap="cool")


if __name__ == "__main__":
    bulk_coverage_scene()
