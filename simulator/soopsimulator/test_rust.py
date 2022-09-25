from rust_sim_core import find_specular_points

import numpy as np

receivers = np.zeros((5, 2, 3))
transmitters = np.ones((5, 3, 3))
speculars = find_specular_points(receivers, transmitters)

print(receivers[0, ...])
print(transmitters[0, ...])
print(speculars[0, ...])
