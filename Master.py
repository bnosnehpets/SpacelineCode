#  PROGRAM DESCRIPTION~|| Simulates motion of a breaking spaceline
#  VERSION~            || v1.1
#  LAST EDITED~        || 06/02/2020

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import matplotlib

# PHYSICAL CONSTANTS
earth_moon_dist = 3.84E8   # 3.84E8 m
earth_mass = 5.972E24      # 5.972E24 kg
moon_mass = 7.348E22       # 7.348E22 kg
G = 6.67E-11               # 6.67E-11 N m^2 kg^-2

# SIMULATION PARAMETERS


# note - to start with - 1 mass in gravitational field of 2 bodies. centrifugal force currently ignored!

def diff(inp, t):
    """
    Input function for 'odeint', takes all r_i and dr_i/dt and gives back derivatives
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param t: Current time in simulation
    :return: The time derivative of the input vector
    """
    r = np.sqrt(inp[0]**2+inp[1]**2)
    a_0 = -G*earth_mass*inp[0]/(r**3) - G*moon_mass*(inp[0]-earth_moon_dist)/((r-earth_moon_dist)**3)
    a_1 = -G*earth_mass*inp[1]/(r**3) - G*moon_mass*(inp[1]-earth_moon_dist)/((r-earth_moon_dist)**3)
    ret = [inp[2],inp[3],a_0, a_1]
    return ret


def simulate():
    """
    Core simulation routine, returns the time-integrated solution (trajectory) by calling odeint
    :return: Vector containing position and velocity of each mass at each time
    """
    init = [1.92E8, 0, 0, 1440.4]            # would be circular orbit if moon wasn't there
    domain = np.linspace(0, 837700*2, 10000) # set to about 2 orbits
    solution = odeint(diff, init, domain)
    return solution