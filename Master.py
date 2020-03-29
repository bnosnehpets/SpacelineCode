#  PROGRAM DESCRIPTION~|| Simulates motion of a breaking spaceline
#  VERSION~            || 2.1
#  LAST EDITED~        || 29/03/2020

# TODO don't allow to go through moon or earth
# TODO make sure don't have any compression!      --- PLUS effect on jacobian
# TODO get rid of ALL loops in main simulation part of program

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# PHYSICAL CONSTANTS
tem = 27.322*24*3600        # Earth-moon rotation period around CoM

omega = 2*np.pi/tem         # Earth-moon angular velocity around CoM
earth_moon_dist = 3.84E8    # 3.84E8 m
earth_mass = 5.972E24       # 5.972E24 kg
moon_mass = 7.348E22        # 7.348E22 kg
earth_radius = 6371000      # 6371000 kg
moon_radius = 1731100       # 1731100 m
G = 6.67E-11                # 6.67E-11 N m^2 kg^-2
com_distance = earth_moon_dist * moon_mass / (earth_mass + moon_mass)

# SIMULATION PARAMETERS
k_over_m = 1E-10*1000000    # 1E-10 s^-2
starting_percentage = 0.75  # For init
no_masses = 80
natural_length = (starting_percentage * earth_moon_dist - moon_radius) / (no_masses - 1)

breaking_strain = 0.05      # Strain (extension/natural_length) at which a spring breaks
breaks = [0] + [35] + [no_masses]   # Index of first mass of each new section of wire (updated during simulation)
repel_force_mag = 10

# PROGRAM PARAMETERS (updated as program runs)
update_count = 0
jac_count = 0


# ANIMATION FUNCTIONS
def animate(sol, title):
    """
    Produces an mp4 file, saved in the current directory
    :param sol: The solution as returned from the odeint function
    :param title: The title seen in the animation video
    :return: void
    """

    def update_frame(num, dat, dot):
        dot.set_data(dat[num, :, :])

        return dot,

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    a = np.transpose(sol[:no_masses*2, :])
    b = a[:, 0::2]
    c = a[:, 1::2]
    d = np.append(b, c, axis=1)
    data = np.reshape(d, (len(a), 2, no_masses))

    fig = plt.figure()
    l, = plt.plot([], [], 'ko', markersize=1)
    plt.title(title)

    # Add earth and moon and set up plot
    plt.xlim(-earth_moon_dist*0.75, earth_moon_dist*1.25)       #   -earth_radius*4 - com_distance, earth_radius*4 - com_distance
    plt.ylim(-earth_moon_dist, earth_moon_dist)       #  -earth_radius*4, earth_radius*4
    earth = plt.Circle((-com_distance, 0), earth_radius, color='b')
    moon = plt.Circle((earth_moon_dist - com_distance, 0), moon_radius, color='gray')
    plt.gcf().gca().add_artist(earth)
    plt.gcf().gca().add_artist(moon)

    ani = animation.FuncAnimation(fig, update_frame, frames=len(data), fargs=(data, l), interval=2, blit=True)

    #  plt.show()
    ani.save('ani_ivp1.mp4', writer=writer, dpi=300)
    #  plt.clf()


# SIMULATION SET-UP FUNCTIONS
def init_conditions():
    """
    :return: Vector of initial conditions of form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    """

    def next_it(rs):
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
        new = next_it(sol[i])
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


# MAIN SIMULATION FUNCTIONS
def update_breaks(mod_diffs, t):
    """
    Updates global 'breaks' parameter based on breaking strain. Ancillary function for update().
    :param mod_diffs: Distance between masses (1st value is between 0th and 1st mass). Note that each value is repeated
    due to use in 'update' function. Final two values should be ignored as they are the distance between the final and
    0th mass.
    :param t: Current time in simulation
    :return None
    """
    nums = []
    for j in range(no_masses - 1):
        if mod_diffs[2*j] > natural_length*(1 + breaking_strain):
            nums.append(j + 1)

    global breaks
    breaks = nums + breaks
    breaks.sort()
    breaks = list(dict.fromkeys(breaks))  # Removes duplicate values

    return None


def update(t, inp):
    """
    Input function for 'odeint', takes all r_i and dr_i/dt and gives back derivatives
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param t: Current time in simulation
    starts (apart from the final one)
    :return: The time derivative of the input vector
    """
    global update_count
    update_count += 1
    print(t)

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

    # Gravity and repelling force
    a = -G * earth_mass * r_from_earth / (mod_r_e ** 3) - G * moon_mass * r_from_moon / (mod_r_m ** 3)

    # Centrifugal
    a += omega ** 2 * positions

    # Coriolis
    tmp = np.zeros(len(velocities))
    tmp[::2] = -velocities[1::2]
    tmp[1::2] = velocities[::2]
    a += 2 * omega * tmp

    # Springs
    p1 = np.copy(positions)
    p2 = np.roll(p1, -2)
    diffs = p2 - p1
    tmp = diffs ** 2
    mod_diffs = np.sqrt(np.repeat([sum(tmp[i:i + 2]) for i in range(0, len(tmp), 2)], 2))
    update_breaks(mod_diffs, t)

    for i in range(len(breaks) - 1):
        # to the right
        p1 = np.copy(positions[2*breaks[i]:2*breaks[i + 1]])
        p2 = np.roll(p1, -2)
        diffs = p2 - p1
        tmp = diffs**2
        mod_diffs = np.sqrt(np.repeat([sum(tmp[i:i + 2]) for i in range(0, len(tmp), 2)], 2))
        extensions = diffs*(1 - np.repeat(natural_length, len(diffs))/mod_diffs)
        extensions[-2:] = np.array([0, 0])  # final mass has nothing to the other side (gets rolled to first mass)
        a[2*breaks[i]:2*breaks[i + 1]] += k_over_m * extensions
        # to the left
        a[2*breaks[i]:2*breaks[i + 1]] += -k_over_m * (np.roll(extensions, 2))

    # Keep one mass at moon
    velocities[-2] = 0
    a[-2] = 0
    velocities[-1] = 0
    a[-1] = 0

    # Combine velocities and accelerations
    ret = np.concatenate([velocities, a])
    return ret


def j_func(t, inp):
    """
    Returns Jacobian matrix for odeint. Form (of partial derivatives) is:
    [[d(dx_0/dt)/dx_0, d(dy_0/dt)/dx_0,                 ..., d(d^2x_0/dt^2)/dx_0, d(d^2y_0/dt^2)/dx_0,                 ...],
     [d(dx_0/dt)/dy_0, d(dy_0/dt)/dy_0,                 ..., d(d^2x_0/dt^2)/dy_0, d(d^2y_0/dt^2)/dy_0,                 ...],
     .
     .
     .
     [d(dx_0/dt)/d(dx_n-1/dt), d(dy_0/dt)/d(dx_n-1/dt), ..., d(d^2x_0/dt^2)/d(dx_n-1/dt), d(d^2y_0/dt^2)/d(dx_n-1/dt), ...],
     [d(dx_0/dt)/d(dy_n-1/dt), d(dy_0/dt)/d(dy_n-1/dt), ..., d(d^2x_0/dt^2)/d(dy_n-1/dt), d(d^2y_0/dt^2)/d(dy_n-1d/t), ...]]
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param t: Current time in simulation
    :return: Jacobian matrix for odeint() (as above)
    """
    global jac_count
    jac_count += 1

    jac = np.zeros((4 * no_masses, 4 * no_masses))
    rs = inp[:2 * no_masses]
    vs = inp[2 * no_masses:]

    # 1. velocities by positions (all 0)

    # 2. velocities by velocities
    dv_dv = jac[2 * no_masses:, :2 * no_masses]
    np.fill_diagonal(dv_dv, 1)
    dv_dv[-1, -1] = 0

    # 3. accelerations by velocities
    da_dv = jac[2 * no_masses:, 2 * no_masses:]
    top_right = da_dv[:-1, 1:]
    bottom_left = da_dv[1:, :-1]
    np.fill_diagonal(top_right[::2, ::2], 2 * omega)
    np.fill_diagonal(bottom_left[::2, ::2], -2 * omega)

    # 4. accelerations by positions
    da_dr = jac[:2 * no_masses, 2 * no_masses:]

    # Gravity part
    def f1(mass, p1, p2):
        return G * mass * (2 * p1 ** 2 - p2 ** 2) / np.hypot(p1, p2) ** 5

    def f2(mass, p1, p2):
        return -3 * G * mass * p1 * p2 / np.hypot(p1, p2) ** 5

    rs_shaped = np.tile(rs, no_masses).reshape(no_masses, 2 * no_masses)
    xs = rs_shaped[:, ::2]
    xs_earth = xs + com_distance
    xs_moon = xs + com_distance - earth_moon_dist
    ys = rs_shaped[:, 1::2]

    # Spring part
    mk = np.zeros((no_masses, no_masses))
    np.fill_diagonal(mk, -2 * k_over_m)
    np.fill_diagonal(mk[:-1, 1:], k_over_m)
    np.fill_diagonal(mk[1:, :-1], k_over_m)
    diag = np.diagonal(mk)
    diag.flags.writeable = True
    upper_diag = np.diagonal(mk[:-1, 1:])
    upper_diag.flags.writeable = True
    lower_diag = np.diagonal(mk[1:, :-1])
    lower_diag.flags.writeable = True

    # first mass (last mass sorted at the end)
    diag[0] += k_over_m

    # all other masses
    for i in range(len(breaks) - 2):
        diag[breaks[i+1] - 1] += k_over_m
        lower_diag[breaks[i+1] - 1] = 0
    for i in range(len(breaks) - 2):
        diag[breaks[i+1]] += k_over_m
        upper_diag[breaks[i+1] - 1] = 0

    # da_x/dx
    da_dr[::2, ::2] = omega ** 2 + f1(earth_mass, xs_earth, ys) + f1(moon_mass, xs_moon, ys) + mk
    # da_x/dy
    da_dr[1::2, ::2] = omega ** 2 + f2(earth_mass, xs_earth, ys) + f2(moon_mass, xs_moon, ys)
    # da_y/dx
    da_dr[::2, 1::2] = da_dr[1::2, ::2]
    # da_y/dy
    da_dr[1::2, 1::2] = omega ** 2 + f1(earth_mass, ys, xs_earth) + f1(moon_mass, ys, xs_moon) + mk

    # Final mass:
    jac[:, -2:] = 0

    return np.transpose(jac)


def simulate(initial, dom):
    """
    Core simulation routine, returns the time-integrated solution (trajectory) by calling odeint
    :init: initial conditions vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :return: Vector containing position and velocity of each mass at each time
    """
    t1 = time.time()
    r = solve_ivp(update, (dom[0], dom[-1]), initial, t_eval=dom, jac=j_func)
    t2 = time.time()
    print(t2-t1)
    return r.y


# MAIN
domain = np.linspace(0, 50000, 1000)   # 837700*2  TODO: THIS IS CRAP NUMBERS
init = init_conditions()   # TESTER: np.concatenate([[3*earth_radius, -0.4*earth_radius], init_conditions(), [0, 0]])
solution = simulate(init, domain)
print(update_count)
print(jac_count)
animate(solution, ":D <3 <3")



# # FOR TESTING JACOBIAN
# no_masses = 2
# com_distance = 1
# earth_moon_dist = 4
# earth_mass = 0.5
# moon_mass = 0.25
# k_over_m = 1
# G = 1
# omega = 1
# print(j_func(np.array([0, 0, 2, 0, 0, 1, 0, -1]), 1))
