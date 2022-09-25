from simulator.soopsimulator.constellations import (
    Constellation,
    ConstellationCollection,
    PropagationConfig,
    OrbitDefinition,
)

from python_sim_core.specular import find_specular_points as find_specular_points_py
from python_sim_core.specular_numba import (
    find_specular_points as find_specular_points_nb,
)
from rust_sim_core import find_specular_points as find_specular_points_rs
import time

prop_config = PropagationConfig(t_step=1.0 / (24.0 * 60.0), t_range=30.0)

# Use 5 L-band transmitter constellations
transmitter_constellations = [
    Constellation.from_tle("data/TLE/gps.txt", None),
    Constellation.from_tle("data/TLE/glonass.txt", None),
    Constellation.from_tle("data/TLE/galileo.txt", None),
    Constellation.from_tle("data/TLE/beidou.txt", None),
    Constellation.from_tle("data/TLE/iridium.txt", None),
]
transmitter_collections = ConstellationCollection(transmitter_constellations)
transmitter_positions = transmitter_collections.propagate_orbits(prop_config)

# Use a constellation of 8 receiver satellites
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

receiver_constellations = [Constellation.from_orbit_definitions(receiver_orbits, None)]
receiver_collections = ConstellationCollection(receiver_constellations)
receiver_positions = receiver_collections.propagate_orbits(prop_config)

# Run 1: 154
# Run 2:
start = time.time()
specular_py = find_specular_points_py(
    transmitter_positions,  # noqa
    receiver_positions,  # noqa
)
print("Numpy: %.2f (s)" % (time.time() - start))
del specular_py

# Run 1: 33.4
# Run 2:
start = time.time()
specular_nb = find_specular_points_nb(
    transmitter_positions,  # noqa
    receiver_positions,  # noqa
)
print("Numba: %.2f (s)" % (time.time() - start))
del specular_nb

# Run 1: 5.92
# Run 2:
start = time.time()
specular = find_specular_points_rs(
    transmitter_positions,  # noqa
    receiver_positions,  # noqa
)
print("Rust: %.2f (s)" % (time.time() - start))
del specular
