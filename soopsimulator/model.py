from constellations import ConstellationCollection, PropagationConfig

from specular import find_specular_points


# TODO: Set up chunking:
#       break out calculate_Specular points and
# #     filter_by_angle constraints into some other class?
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

    def filter_by_angle_constraints(self):
        # Get the antenna cone angles
        (
            tran_nadir_angles,
            tran_zenith_angles,
        ) = self.transmitter_constellations.get_angles()

        (
            recv_nadir_angles,
            recv_zenith_angles,
        ) = self.receiver_constellations.get_angles()
