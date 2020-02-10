#  PROGRAM DESCRIPTION~|| Simulates motion of a breaking spaceline
#  VERSION~            || v1.3
#  LAST EDITED~        || 10/02/2020

#note currently doesnt quite work.... need to check diff function

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import matplotlib

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

# SIMULATION PARAMETERS
mass = 100
k = 100

def init_basic(n, l, i):
    """
    Sets up input vector of the form [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt], such that each spring is
    un-extended and the line stretches l % of the way from the moon to the Earth. The line is cut between the ith and
    i+1th mass
    Note that in reality the line should be under tension and hence these initial conditions are wrong
    :param n: number of masses
    :param l: percentage of way the nth mass is from the moon centre to the Earth centre
    :param i: the line is cut between the ith and i+1th mass
    :return: vector as described above
    """
    ret = np.zeros(n*4)
    for j in range(n):
        ret[4*j] = earth_moon_dist - moon_radius - com_distance - i*(earth_moon_dist*l - moon_radius)/(n-1)
        # all others remain 0


def diff(inp, t):
    """
    Input function for 'odeint', takes all r_i and dr_i/dt and gives back derivatives
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param t: Current time in simulation
    :return: The time derivative of the input vector
    """

    positions = inp[:len(inp)//2]
    velocities = inp[len(inp)//2:]

    # distance to earth (correctly shaped)
    r_from_earth = np.copy(positions)
    r_from_earth[::2] += com_distance
    tmp1 = r_from_earth ** 2
    r_e = np.sqrt(np.repeat([sum(tmp1[i:i+2]) for i in range(0, len(tmp1), 2)], 2))

    # distance to moon (correctly shaped)
    r_from_moon = np.copy(positions)
    r_from_moon[::2] += com_distance - earth_moon_dist
    tmp1 = r_from_moon ** 2
    r_m = np.sqrt(np.repeat([sum(tmp1[i:i+2]) for i in range(0, len(tmp1), 2)], 2))

    # Gravity
    a = -G*earth_mass*r_from_earth/(r_e**3) - G*moon_mass*r_from_moon/(r_m**3)

    # Centrifugal
    a += omega**2 * positions

    # Coriolis
    tmp = np.zeros(len(velocities))
    tmp[::2] = -velocities[1::2]
    tmp[1::2] = velocities[::2]
    a += 2 * omega * tmp

    # Combine velocities and accelerations
    ret = np.concatenate([velocities, a])
    return ret


def simulate(init, domain):
    """
    Core simulation routine, returns the time-integrated solution (trajectory) by calling odeint
    :init: initial conditions vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :return: Vector containing position and velocity of each mass at each time
    """
    return odeint(diff, init, domain, full_output=True)


# simulation
domain = np.linspace(0, 837700*2, 10000)
init = [1.92E8-com_distance, 0, 0, 1440.4+omega*(1.92E8-com_distance)]  # would be circular orbit if moon wasn't there
solution = simulate(init, domain)

# plot
x0 = []
x1 = []
for i in range(len(solution[0])):
    x0.append(solution[0][i][0])
    x1.append(solution[0][i][1])
plt.plot(x0, x1, color='k')

# add earth and moon (centred at com)
earth = plt.Circle((-com_distance, 0), earth_radius, color='b')
moon = plt.Circle((earth_moon_dist - com_distance, 0), moon_radius, color='gray')
plt.gcf().gca().add_artist(earth)
plt.gcf().gca().add_artist(moon)
plt.xlim(-earth_moon_dist*0.75, earth_moon_dist*1.25)
plt.show()