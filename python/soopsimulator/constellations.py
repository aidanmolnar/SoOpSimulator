from typing_extensions import Self
from skyfield.api import load
from skyfield.positionlib import Geocentric
from skyfield.framelib import itrs
from skyfield.timelib import Time
from skyfield.sgp4lib import EarthSatellite, Satrec

import numpy as np

from dataclasses import dataclass


class AntennaConfiguration:
    pass


@dataclass
class NadirPointing(AntennaConfiguration):
    zenith_cone_angle: float  # (deg)
    nadir_cone_angle: float  # (deg)


@dataclass
class OrbitDefinition:
    # Keplerian orbit elements
    e: float  # Eccentricity
    a: float  # Semimajor axis (km)
    i: float  # Inclination (deg)
    RAAN: float  # Right argument of the ascending node (deg)
    AOP: float  # Argument of periapsis (deg)
    TA: float  # True anomaly (deg)

    # Other TLE elements
    bstar: float = 0.0  # drag coefficient (/earth radii)
    epoch: float = 0.0  # epoch: days since 1949 December 31 00:00 UT

    # Output in radians
    def mean_anomaly(self) -> float:
        E = 2.0 * np.arctan(
            np.sqrt((1.0 - self.e) / (1.0 + self.e)) * np.tan(np.deg2rad(self.TA) / 2.0)
        )
        return E - self.e * np.sin(E)

    # Output in radians/s
    def mean_motion(self):
        mu = 3.986004418e5  # Earth Gravitational Parameter(km^3-s)
        return np.sqrt(mu / np.power(self.a, 3.0))


# A group of satellites with the same antenna configuration or from the same TLE
class Constellation:
    def __init__(
        self,
        satellites: list[EarthSatellite],
        antenna_configuration: AntennaConfiguration,
    ) -> None:
        self.satellites = satellites
        self.antenna_configuration = antenna_configuration

    def from_tle(
        tle_path: str,
        antenna_configuration: AntennaConfiguration,
    ):
        return Constellation(load.tle_file(tle_path), antenna_configuration)

    def from_orbit_definitions(
        orbit_definitions: list[OrbitDefinition],
        antenna_configuration: AntennaConfiguration,
    ):
        satellites = []
        for i, orbit in enumerate(orbit_definitions):
            # Define satellite using elements directly
            satrec = Satrec()
            satrec.sgp4init(
                2,  # Corresponds to WGS84 gravity model
                "i",  # 'a' = old AFSPC mode, 'i' = improved mode
                i,  # satnum: Satellite number
                orbit.epoch,  # epoch: days since 1949 December 31 00:00 UT
                orbit.bstar,  # bstar: drag coefficient (/earth radii)
                0.0,  # ndot: ballistic coefficient (revs/day) (unused)
                0.0,  # nddot: second derivative of mean motion (revs/day^3) (unused)
                orbit.e,  # ecco: eccentricity
                np.deg2rad(orbit.AOP),  # argpo: argument of perigee (radians)
                np.deg2rad(orbit.i),  # inclo: inclination (radians)
                np.deg2rad(orbit.mean_anomaly()),  # mo: mean anomaly (radians)
                60 * orbit.mean_motion(),  # no_kozai: mean motion (radians/minute)
                np.deg2rad(orbit.RAAN),  # nodeo: right ascension (radians)
            )
            satellites.append(EarthSatellite.from_satrec(satrec, load.timescale()))

        return Constellation(satellites, antenna_configuration)


@dataclass
class PropagationConfig:
    t_step: float  # Time between orbit sample points (days)
    t_range: float  # How long to simulate for (days)

    def offsets(self, t_middle) -> Time:
        return t_middle + np.arange(
            -self.t_range / 2.0, self.t_range / 2.0, self.t_step
        )


class ConstellationCollection:
    def __init__(
        self,
        constellations: list[Constellation],
    ) -> Self:
        self.constellations = constellations
        self.ts = load.timescale()

    # List of all satellites in the collection
    def get_satellites(self) -> list[EarthSatellite]:
        satellites = []
        for constellation in self.constellations:
            satellites.extend(constellation.satellites)
        return satellites

    def get_average_epoch_tai(satellites: list[EarthSatellite]) -> float:
        epochs = np.zeros(len(satellites))
        for i, satellite in enumerate(satellites):
            # Use universal time scale (International Atomic Time)
            epochs[i] = satellite.epoch.tai
        return np.average(epochs)

    # Get IRTS coordinates of satellites over time samples defined by propagation config
    # Returns a numpy array of (len(time samples), len(satellites) , 3)
    # Last dimension is x,y,z in IRTS
    def propagate_orbits(self, propagation_config: PropagationConfig) -> np.ndarray:
        satellites = self.get_satellites()

        t_middle = ConstellationCollection.get_average_epoch_tai(satellites)
        t = self.ts.tai_jd(
            propagation_config.offsets(t_middle),
        )

        pos_arrays = []
        for satellite in satellites:
            geocentric: Geocentric = satellite.at(t)

            # A len(times) by 3 array of ITRS positions
            pos = geocentric.frame_xyz(itrs).km.transpose()
            pos_arrays.append(pos)

        return np.stack(pos_arrays, axis=1)
