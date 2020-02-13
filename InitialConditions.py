#  PROGRAM DESCRIPTION~|| Calculates initial conditions of stable spaceline via iterative procedure
#  VERSION~            || v1.1
#  LAST EDITED~        || 11/02/2020

import numpy as np

# PHYSICAL CONSTANTS
tem = 27.322*24*3600       # Earth-moon rotation period around CoM
omega = 2*np.pi/tem        # Earth-moon angular velocity around CoM
earth_moon_dist = 3.84E8   # 3.84E8 m
earth_mass = 5.972E24      # 5.972E24 kg
moon_mass = 7.348E22       # 7.348E22 kg
earth_radius = 6371000     # 6371000 kg
moon_radius = 1731100      # 1731100 m
G = 6.67E-11               # 6.67E-11 N m^2 kg^-2
com_distance = earth_moon_dist * moon_mass / (earth_mass + moon_mass)

k_over_m = 1E-10*1000000       # 1E-10 s^-2


def update(rs):
    """
    Performs iteration of calculation to find stable initial conditions
    :param rs: Array of distance of each mass from moon (from previous iteration)
    :return: Array of distance of each mass from moon (next iteration)
    """
    rs_moon = rs + np.repeat(moon_radius, no_masses)
    rs_earth = np.repeat(earth_moon_dist, no_masses) - rs_moon
    rs_com = rs_earth - np.repeat(com_distance, no_masses)

    accelerations = G * earth_mass / (rs_earth**2) - G * moon_mass / (rs_moon**2) + omega**2 * rs_com
    extensions = (1/k_over_m) * np.flip(np.cumsum(np.flip(accelerations)))
    a = np.cumsum(np.repeat(diff, no_masses))
    rs_updated = a + np.cumsum(extensions)
    return rs_updated


starting_percentage = 0.5
no_masses = 100

#  Initial
diff = (starting_percentage * earth_moon_dist - moon_radius) / no_masses  # also natural length
rs_init = np.cumsum(np.repeat(diff, no_masses))
solution = [rs_init]

# Run iterations
r = 9
for i in range(r):
    new = update(solution[i])
    solution = np.reshape(np.append(solution, new), (i+2, len(solution[0])))

# Convert into main program coordinate system
final = np.flip(np.repeat(earth_moon_dist - com_distance - moon_radius, no_masses) - solution[-1])
print(final)
