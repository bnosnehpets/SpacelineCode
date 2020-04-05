#  PROGRAM DESCRIPTION~|| Simulates motion of a breaking spaceline
#  VERSION~            || 2.1.1
#  LAST EDITED~        || 05/04/2020

# TODO don't allow to go through moon or earth - IN PROCESS
# TODO match Jacobian to Earth/moon contact (not urgent as solve_ivp does not seem to use it).
# TODO make sure don't have any compression. (Plus effect on Jacobian, as above)
# TODO get rid of ALL loops in main simulation part of program.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from functools import partial

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
repel_force_mag = 10

# USEFUL FOR REPEATED USE
zeros = np.zeros((no_masses,))


# ANIMATION FUNCTIONS
def animate(sol, title, save_file, file_name):
    """
    Produces an mp4 file, saved in the current directory
    :param sol: The solution as returned from the odeint function
    :param title: The title seen in the animation video
    :param save_file: True to save mp4, False to see animation
    :param file_name: Name of file (plus .mp4)
    :return: void
    """

    def update_frame(num, dat, dot):
        dot.set_data(dat[num, :, :])

        return dot,

    if save_file:
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
    plt.xlim(-earth_radius*4 - com_distance, earth_radius*4 - com_distance)       #  -earth_moon_dist*0.75, earth_moon_dist*1.25
    plt.ylim(-earth_radius*4, earth_radius*4)       #  -earth_moon_dist, earth_moon_dist
    earth = plt.Circle((-com_distance, 0), earth_radius, color='b')
    moon = plt.Circle((earth_moon_dist - com_distance, 0), moon_radius, color='gray')
    plt.gcf().gca().add_artist(earth)
    plt.gcf().gca().add_artist(moon)

    ani = animation.FuncAnimation(fig, update_frame, frames=len(data), fargs=(data, l), interval=2, blit=True)

    if save_file:
        ani.save(file_name+'.mp4', writer=writer, dpi=300)
    else:
        plt.show()
    plt.clf()


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
    Updates global 'gBreaks' parameter based on breaking strain. Ancillary function for update().
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

    global gBreaks
    gBreaks = nums + gBreaks
    gBreaks.sort()
    gBreaks = list(dict.fromkeys(gBreaks))  # Removes duplicate values

    return None


def earth_crash_event(t, inp, tot_t, i):
    """

    :param t: Current time in simulation
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param tot_t: Total time in simulation (including previous calls of solve_ivp)
    :param i: Index for mass
    :return:
    """
    rx = inp[2 * i] + com_distance
    ry = inp[2 * i + 1]
    ret = np.sqrt(rx ** 2 + ry ** 2) - earth_radius
    if ret < 0:
        print("Mass "+str(i)+" has crashed into Earth: r-R = "+str(ret))
    return ret


def moon_crash_event(t, inp, tot_t, i):
    """

    :param t: Current time in simulation
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param tot_t: Total time in simulation (including previous calls of solve_ivp)
    :param i: Index for mass
    :return:
    """
    rx = inp[2 * i] + com_distance - earth_moon_dist
    ry = inp[2 * i + 1]
    ret = np.sqrt(rx ** 2 + ry ** 2) - moon_radius
    if ret < 0:
        print("Mass "+str(i)+" has crashed into Earth: r-R = "+str(ret))
    return ret


def acc_towards(rs, accs, earth):
    """
    Tests whether the acceleration is towards the Earth/moon for masses in contact with Earth/moon
    :param rs: Position vectors centred at Earth/moon
    :param accs: Acceleration vectors
    :param earth: True if testing Earth, False if testing moon
    :return: Array of booleans, True if above two conditions satisfied, False if not
    """
    contact = gEarth_contact if earth else gMoon_contact
    r_angles = np.where(contact, [np.arctan2(rs[i + 1], rs[i]) for i in range(0, 2 * (no_masses - 1), 2)], 0)
    axs = accs[:-2:2]
    ays = accs[1:-2:2]

    a_perp = np.where(contact, axs*np.cos(r_angles) + ays*np.sin(r_angles), 0)
    ret = np.logical_and(contact, a_perp < 0)
    return ret


def j_func(t, inp, tot_t):
    """
    Returns Jacobian matrix for odeint. Form (of partial derivatives) is:
    [[d(dx_0/dt)/dx_0, d(dy_0/dt)/dx_0,                 ..., d(d^2x_0/dt^2)/dx_0, d(d^2y_0/dt^2)/dx_0,                 ...],
     [d(dx_0/dt)/dy_0, d(dy_0/dt)/dy_0,                 ..., d(d^2x_0/dt^2)/dy_0, d(d^2y_0/dt^2)/dy_0,                 ...],
     .
     .
     .
     [d(dx_0/dt)/d(dx_n-1/dt), d(dy_0/dt)/d(dx_n-1/dt), ..., d(d^2x_0/dt^2)/d(dx_n-1/dt), d(d^2y_0/dt^2)/d(dx_n-1/dt), ...],
     [d(dx_0/dt)/d(dy_n-1/dt), d(dy_0/dt)/d(dy_n-1/dt), ..., d(d^2x_0/dt^2)/d(dy_n-1/dt), d(d^2y_0/dt^2)/d(dy_n-1d/t), ...]]
    :param t: Current time in simulation
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param tot_t: Total time in simulation (including previous calls of solve_ivp)
    :return: Jacobian matrix for odeint() (as above)
    """
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
    for i in range(len(gBreaks) - 2):
        diag[gBreaks[i+1] - 1] += k_over_m
        lower_diag[gBreaks[i+1] - 1] = 0
    for i in range(len(gBreaks) - 2):
        diag[gBreaks[i+1]] += k_over_m
        upper_diag[gBreaks[i+1] - 1] = 0

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


def update(t, inp, tot_t):
    """
    Input function for 'odeint', takes all r_i and dr_i/dt and gives back derivatives
    :param t: Current time in simulation
    :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :param tot_t: Total time in simulation (including previous calls of solve_ivp)
    :return: The time derivative of the input vector
    """
    print(t+tot_t)

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

    for i in range(len(gBreaks) - 1):
        # to the right
        p1 = np.copy(positions[2*gBreaks[i]:2*gBreaks[i + 1]])
        p2 = np.roll(p1, -2)
        diffs = p2 - p1
        tmp = diffs**2
        mod_diffs = np.sqrt(np.repeat([sum(tmp[i:i + 2]) for i in range(0, len(tmp), 2)], 2))
        extensions = diffs*(1 - np.repeat(natural_length, len(diffs))/mod_diffs)
        extensions[-2:] = np.array([0, 0])  # final mass has nothing to the other side (gets rolled to first mass)
        a[2 * gBreaks[i]: 2 * gBreaks[i + 1]] += k_over_m * extensions
        # to the left
        a[2 * gBreaks[i]: 2 * gBreaks[i + 1]] += -k_over_m * (np.roll(extensions, 2))

    # Keep one mass at moon
    velocities[-2] = 0
    a[-2] = 0
    velocities[-1] = 0
    a[-1] = 0

    #  Set accelerations of masses in contact with Earth or moon and not being pulled away to 0
    rce = acc_towards(r_from_earth, a, True)
    global gEarth_contact
    TMP = np.copy(gEarth_contact)  # TMP FOR TESTING
    gEarth_contact = np.where(gEarth_contact,
                              np.where(np.logical_not(rce),
                                       [False if events[i](t, inp, tot_t) > 0 else True for i in range(len(events)//2)],
                                       gEarth_contact),
                              gEarth_contact)
    a[:-2] = np.where(np.repeat(gEarth_contact, 2), np.where(np.repeat(rce, 2), 0, a[:-2]), a[:-2])
    if np.any(np.logical_and(TMP, np.logical_not(gEarth_contact))):  # FOR TESTING
        print("The mass has left the surface: r-R = "+str(mod_diffs[0]))

    rcm = acc_towards(r_from_moon, a, False)
    global gMoon_contact
    gMoon_contact \
        = np.where(gMoon_contact,
                   np.where(np.logical_not(rcm),
                            [False if events[len(events)//2 + i](t, inp, tot_t) > 0 else True for i in range(len(events)//2)],
                            gMoon_contact),
                   gMoon_contact)
    a[:-2] = np.where(np.repeat(gEarth_contact, 2), np.where(np.repeat(rce, 2), 0, a[:-2]), a[:-2])

    # Combine velocities and accelerations
    ret = np.concatenate([velocities, a])
    return ret


def simulate(initial, dom):
    """
    Core simulation routine, returns the time-integrated solution by calling solve_ivp (several times)
    :init: initial conditions vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
    :return: Vector containing position and velocity of each mass at each time
    """
    t1 = time.time()

    sol = np.array([], dtype=np.int64).reshape(4 * no_masses, 0)
    f_calls = 0
    j_calls = 0
    t_tot = 0

    while t_tot < dom[-1]:
        r = solve_ivp(update, (dom[0], dom[-1]), initial, t_eval=dom, jac=j_func, events=events, args=(t_tot,))

        # Update overall variables
        sol = np.hstack([sol, r.y])
        f_calls += r.nfev
        j_calls += r.njev

        if r.status == 0:
            t_tot = dom[-1]

        # If not finished update conditions for next run of solve_ivp
        if r.status != 0:
            print(len(r.t_events))
            print(len(r.y_events))
            earth, index, value, new_init = [(i < no_masses, i % no_masses, r.t_events[i][0], r.y_events[i].flatten())
                                             for i in range(len(r.t_events)) if r.t_events[i].size != 0][0]
            t_tot += r.t_events[index][0]
            initial = new_init
            initial[2 * (no_masses + index): 2 * (no_masses + index + 1)] = 0
            if earth:
                gEarth_contact[index] = True
            else:
                gMoon_contact[index] = True

    t2 = time.time()
    print(t2 - t1)
    return sol, f_calls, j_calls


# MAIN
domain = np.linspace(0, 89860, 2000)   # 100000    837700*2
init = init_conditions()

#  Create dictionary of functions from crash_event 'base functions' for usage in solve_ivp, one for each mass. Do not
#  include function for final mass (which is always attached to the moon)
d_e = {f'e_crash_{k}': partial(earth_crash_event, i=k) for k in range(no_masses - 1)}
d_m = {f'm_crash_{k}': partial(moon_crash_event, i=k) for k in range(no_masses - 1)}
events = []
for _, fn in d_e.items():
    events.append(fn)
    fn.terminal = True
    fn.direction = -1
for _, fn in d_m.items():
    events.append(fn)
    fn.terminal = True
    fn.direction = -1

# Below are UPDATED DURING SIMULATION (GLOBAL VARS)
gBreaks = [0] + [35] + [no_masses]   # Index of first mass of each new section of wire
gMoon_contact = np.array([False] * (no_masses - 1))
gEarth_contact = np.array([False] * (no_masses - 1))

solution, f_evals, j_evals = simulate(init, domain)
print(f_evals)
print(j_evals)
animate(solution, "TESTING CRASH FUNCTIONALITY", False, "ani_crash2")
