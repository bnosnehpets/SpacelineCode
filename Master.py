#  PROGRAM DESCRIPTION~|| Simulates motion of a breaking spaceline
#  VERSION~            || v1.4
#  LAST EDITED~        || 14/02/2020

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
k_over_m = 1E-10*1000000   # 1E-10 s^-2
starting_percentage = 0.5  # For init
no_masses = 101
natural_length = (starting_percentage * earth_moon_dist - moon_radius) / (no_masses - 1)


def animate(sol, title):
    """
    Produces an mp4 file, saved in the current directory
    :param sol: The solution as returned from the odeint function
    :param title: The title seen in the animation video
    :return: void
    """

    def update(num, dat, dot):
        dot.set_data(dat[num, :, :])
        return dot,

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()

    a = sol[0][:, :no_masses*2]
    b = a[:, 0::2]
    c = a[:, 1::2]
    d = np.append(b, c, axis=1)
    data = np.reshape(d, (len(a), 2, no_masses))

    l, = plt.plot([], [], 'ko', markersize=0.1)
    plt.title(title)
    # Add earth and moon and set up correctly
    plt.xlim(-earth_moon_dist*0.75, earth_moon_dist*1.25)
    plt.ylim(-earth_moon_dist, earth_moon_dist)

    earth = plt.Circle((-com_distance, 0), earth_radius, color='b')
    moon = plt.Circle((earth_moon_dist - com_distance, 0), moon_radius, color='gray')
    plt.gcf().gca().add_artist(earth)
    plt.gcf().gca().add_artist(moon)

    ani = animation.FuncAnimation(fig, update, len(data), fargs=(data, l), interval=1, blit=True)

    plt.show()
    # ani.save('ani.mp4', writer=writer, dpi=300)

    # clear plot
    plt.clf()


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
    tmp = r_from_earth ** 2
    mod_r_e = np.sqrt(np.repeat([sum(tmp[i:i+2]) for i in range(0, len(tmp), 2)], 2))

    # distance to moon (correctly shaped)
    r_from_moon = np.copy(positions)
    r_from_moon[::2] += com_distance - earth_moon_dist
    tmp1 = r_from_moon ** 2
    mod_r_m = np.sqrt(np.repeat([sum(tmp1[i:i+2]) for i in range(0, len(tmp1), 2)], 2))

    # Gravity
    a = -G*earth_mass*r_from_earth/(mod_r_e**3) - G*moon_mass*r_from_moon/(mod_r_m**3)

    # Centrifugal
    a += omega**2 * positions

    # Coriolis
    tmp = np.zeros(len(velocities))
    tmp[::2] = -velocities[1::2]
    tmp[1::2] = velocities[::2]
    a += 2 * omega * tmp

    # Springs
    # to the right
    p1 = np.copy(positions)
    p2 = np.roll(p1, -2)
    diffs = p2 - p1
    tmp = diffs**2
    mod_diffs = np.sqrt(np.repeat([sum(tmp[i:i + 2]) for i in range(0, len(tmp), 2)], 2))
    extensions = diffs*(1 - np.repeat(natural_length, len(diffs))/mod_diffs)
    extensions[len(extensions) - 2:] = [0, 0]  # final mass has nothing to the other side (gets rolled to first mass)
    a += k_over_m * extensions
    # to the left
    a += -k_over_m * (np.roll(extensions, 2))

    # Keep one mass at moon
    velocities[-1] = 0
    a[-1] = 0

    # Combine velocities and accelerations
    ret = np.concatenate([velocities, a])
    return ret


def simulate(initial, dom):
    """
    Core simulation routine, returns the time-integrated solution (trajectory) by calling odeint
    :init: initial conditions vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :return: Vector containing position and velocity of each mass at each time
    """
    return odeint(diff, initial, dom, full_output=True)


def init_conditions():
    """
    :return: Vector of initial conditions of form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    """
    def update(rs):
        """
        Performs iteration of calculation to find stable initial conditions
        :param rs: Array of distance of each mass from moon (from previous iteration)
        :return: Array of distance of each mass from moon (next iteration)
        """

        rs_moon = rs + np.repeat(moon_radius, no_m)
        rs_earth = np.repeat(earth_moon_dist, no_m) - rs_moon
        rs_com = rs_earth - np.repeat(com_distance, no_m)

        accelerations = G * earth_mass / (rs_earth ** 2) - G * moon_mass / (rs_moon ** 2) + omega ** 2 * rs_com
        extensions = (1 / k_over_m) * np.flip(np.cumsum(np.flip(accelerations)))
        a = np.cumsum(np.repeat(natural_length, no_m))
        rs_updated = a + np.cumsum(extensions)
        return rs_updated

    #  Initial
    no_m = no_masses - 1
    rs_init = np.cumsum(np.repeat(natural_length, no_m))
    sol = [rs_init]

    # Run iterations
    r = 20  # TODO: change to check when it is accurate enough
    for i in range(r):
        new = update(sol[i])
        sol = np.reshape(np.append(sol, new), (i + 2, len(sol[0])))

    # Convert into main program coordinate system
    sol_converted = np.flip(np.repeat(earth_moon_dist - com_distance - moon_radius, no_m) - sol[-1])
    # Append mass at moon
    rs_final = np.append(sol_converted, [earth_moon_dist - com_distance - moon_radius])
    # Intersperse with 0s (y, vx, vy)
    positions = [0] * 2 * len(rs_final)
    positions[::2] = rs_final
    velocities = [0] * 2 * len(rs_final)
    final = np.append(positions, velocities)
    return final


# MAIN
domain = np.linspace(0, 837, 10000)   # 837700*2
init = init_conditions()
solution = simulate(init, domain)

animate(solution, ":D <3 <3")
