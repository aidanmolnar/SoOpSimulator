import pytest
import numpy as np
from soopsimulator.constellations import (
    PropagationConfig,
    OrbitDefinition,
    Constellation,
    ConstellationCollection,
)


# Test OrbitDefinition calculations
def test_1():
    # ISS orbit
    orbit = OrbitDefinition(
        e=0.0002204,
        a=6794.0,
        i=51.6425,
        RAAN=257.3620,
        AOP=231.2948,
        TA=20.0,
    )

    # Test values from poliastro
    assert orbit.mean_motion() == pytest.approx(0.00112740660, 1e-8)
    assert orbit.mean_anomaly() == pytest.approx(0.3489151113347133, 1e-8)


# Check output dimensions of single constellation from tle
def test_2():
    collection = ConstellationCollection(
        [Constellation.from_tle("TLE/muos.txt", None)],  # Contains 4 satellites
    )

    t_step = 0.1
    t_range = 4.0

    config = PropagationConfig(t_step, t_range)
    len_t = len(config.offsets(0.0))

    positions = collection.propagate_orbits(config)

    # Length of np.arrange is numerically unstable, check that its close to expected.
    assert len_t == pytest.approx(t_range / t_step, 1.1)
    # Check output has correct shape
    assert positions.shape == (4, len_t, 3)
    # Check all positions are valid
    assert not np.isnan(positions).any()


# Check output dimensions of multiple constellations from orbit definitions
# Check that output positions are not NaN for reasonable orbit (ISS)
def test_3():
    orbits1 = [
        OrbitDefinition(e=0.00, a=6731, i=0.0, RAAN=0.0, AOP=0, TA=0),
        OrbitDefinition(e=0.01, a=6731, i=0.0, RAAN=0.0, AOP=0, TA=0),
    ]
    orbits2 = [
        OrbitDefinition(e=0.02, a=6731, i=0.0, RAAN=0.0, AOP=0, TA=0),
        OrbitDefinition(e=0.03, a=6731, i=0.0, RAAN=0.0, AOP=0, TA=0),
        OrbitDefinition(e=0.04, a=6731, i=0.0, RAAN=0.0, AOP=0, TA=0),
    ]

    collection = ConstellationCollection(
        [
            Constellation.from_orbit_definitions(orbits1, None),
            Constellation.from_orbit_definitions(orbits2, None),
        ],
    )

    t_step = 0.1
    t_range = 30.0

    config = PropagationConfig(t_step, t_range)
    len_t = len(config.offsets(0.0))

    positions = collection.propagate_orbits(config)

    # Length of np.arrange is numerically unstable, check that its close to expected.
    assert len_t == pytest.approx(t_range / t_step, 1.1)
    # Check output has correct shape
    assert positions.shape == (5, len_t, 3)
    # Check all positions are valid
    assert not np.isnan(positions).any()


# Check low altitude satellites (spacebee) are valid over long time frame (1 year)
def test_4():
    collection = ConstellationCollection(
        [Constellation.from_tle("TLE/spacebee.txt", None)],
    )

    t_step = 2.0
    t_range = 180.0

    config = PropagationConfig(t_step, t_range)

    positions = collection.propagate_orbits(config)

    # Check all positions are valid
    assert not np.isnan(positions).any()


# Check circulare quaatorial orbit has low z value and approximately constant radius
def test_5():
    a = 10000

    orbits = [
        OrbitDefinition(e=0.00, a=a, i=0.0, RAAN=0.0, AOP=0, TA=0),
    ]

    collection = ConstellationCollection(
        [
            Constellation.from_orbit_definitions(orbits, None),
        ]
    )

    positions = collection.propagate_orbits(
        PropagationConfig(t_step=0.01, t_range=2.0),
    )

    z = positions[..., 2]
    r = np.linalg.norm(positions[..., 0:2], axis=-1)

    assert (z < 10).all()
    assert (np.logical_and(r > a - 10, r < a + 10)).all()


# Check that geosynchronous satellites don't move very much
# Ideally would be fixed in IRTS frame
def test_6():
    collection = ConstellationCollection(
        [Constellation.from_tle("TLE/skynet.txt", None)],
    )

    positions = collection.propagate_orbits(
        PropagationConfig(t_step=0.25, t_range=5.0),
    )

    # Compute difference from average x and y position for each satellite
    averages = np.average(positions[3:, :, 0:2], axis=1, keepdims=True)
    delta = positions[3:, :, 0:2] - averages

    # Check variations are less than 100km
    assert (np.abs(delta) < 100).all()
