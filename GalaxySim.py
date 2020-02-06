import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import time
import matplotlib

#  PROGRAM      || SIMULATES COLLISION OF 2 GALAXIES. PERFORMS SEVERAL CALCULATIONS AND CREATES SEVERAL PLOTS TO ANALYSE
#  DESCRIPTION~ || THE RESULTING DATA
#  VERSION~     || v4.1
#  LAST EDITED~ || 29/04/2019 08.51


class Simulator:

    #  PROBLEM SETUP/INITIALISATION

    G = 1  # gravitational constant

    def __init__(self, masses, d, max_time, no_steps):
        '''
        Initialises simulation
        :param masses: A list of the mass objects
        :param d: Number of Spatial Dimensions
        :param max_time: Physical time that the simulation ends
        :param no_steps: Number of time steps that the simulator uses
        '''
        # Set up class variables
        self.max_time = max_time
        self.no_steps = no_steps
        self.timestamp = time.time()  # used for file names
        self.solution = np.array([])  # simulate() function fills this with all position and velocity data
        self.d = d
        self.masses = masses
        self.no_masses = len(masses)
        self.mass_vals = []  # stores masses outside object - easier access significantly reduces runtime
        self.mass_indices = []
        self.no_heavies = 0  # Number of heavy masses (non-zero mass)
        self.runtime = 0     # Updated when simulation is done

        #  Orders the masses so that non-zero masses are first and updates the above variables appropriately
        heavies = []
        lights = []
        heavy_m = []
        light_m = []
        heavy_index = []
        light_index = []
        for i in range(len(self.masses)):
            m = self.masses[i]
            if m.asteroid:
                lights.append(m)
                light_m.append(m.mass)
                light_index.append(m.galaxy_index)
            else:
                heavies.append(m)
                heavy_m.append(m.mass)
                self.no_heavies += 1
                heavy_index.append(m.galaxy_index)
        self.masses = np.array(heavies + lights) #  put the heavies at the start of the array
        self.mass_vals = np.array(heavy_m + light_m)
        self.mass_indices = np.array(light_index + heavy_index)

    #  ANCILLARY DATA RETRIEVAL FUNCTIONS

    def pos(self, i, k, t):
        '''
        Calculates single position component for single t value
        :param i: Mass index
        :param k: Cartesian index (e.g. k = 0 returns x value)
        :param t: Time-step
        :return: The position component
        '''
        return self.solution[t, i * self.d + k]

    def dist(self, i, j, t):
        '''
        Calculates distance between mass i and mass j at time t
        :param i: One mass index
        :param j: The other mass index
        :param t: Time-step
        :return: The distance between the two masses
        '''
        square_pos = 0
        for k in range(self.d):
            square_pos += (self.pos(i, k, t) - self.pos(j, k, t)) ** 2
        return np.sqrt(square_pos)

    def pos_hist(self, i, k):
        '''
        Calculates position for all t and one i
        :param i: The mass index
        :param k: Cartesian index (e.g. k = 0 returns x value)
        :return: The position component history as an array
        '''
        hist = []
        for t in range(len(self.solution)):
            hist.append(self.pos(i, k, t))
        return hist

    # WARNING: currently only works in 2D
    def angle(self, i, j, t):
        '''
        Calculates angle between position vectors of mass with index i and mass with index j
        :param i: One mass index
        :param j: The other mass index
        :param t: Time-step
        :return: Angle between the two position vectors
        '''
        return np.arctan((self.pos(i, 1, t) - self.pos(j, 1, t)) / (self.pos(i, 0, t) - self.pos(j, 0, t)))

    def vel(self, i, k, t):
        '''
        Calculates single velocity component for single t value
        :param i: Mass index
        :param k: Cartesian index (e.g. k = 0 returns x value)
        :param t: Time-step
        :return: The velocity component
        '''
        return self.solution[t, self.d*(self.no_masses+i) + k]

    def vel_hist(self, i, k):
        '''
            Calculates position for all t and one i
            :param i: The mass index
            :param k: Cartesian index (e.g. k = 0 returns x value)
            :return: The velocity component history as an array
        '''
        hist = []
        for t in range(len(self.solution)):
            hist.append(self.vel(i, k, t))
        return hist

    #  PLOTTING FUNCTIONS

    def animate_2d(self, title):
        '''
        Produces an mp4 file, saved in the current directory
        :param title: The title seen in the animation video
        :return: void
        '''

        def update(num, data, dot, dot1):
            half = int(self.no_masses / 2)
            dot.set_data(data[num, :, :])
            dot1.set_data(data[num, :, half:])
            return dot, dot1

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        fig1 = plt.figure()

        a = self.solution[:, :self.no_masses*self.d]
        b = a[:, 0::2]
        c = a[:, 1::2]
        d = np.append(b, c, axis=1)
        data = np.reshape(d, (len(a), 2, self.no_masses))

        l, = plt.plot([], [], 'k,', markersize=0.05)
        l1, = plt.plot([], [], 'r,', markersize=0.05)
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.title(title)

        ani = animation.FuncAnimation(fig1, update, len(data), fargs=(data, l, l1), interval=1, blit=True)

        # plt.show()
        ani.save('f_' + str(self.timestamp)+'_ani.mp4', writer=writer, dpi=300)

        # clear plot
        plt.clf()

    #  WARNING: Currently only works in 2D
    def plot_trajectory_2d(self, indices):
        '''
        Plots the trajectory of all masses with index in 'indices' array
        If indices = [-1], then plots all trajectories
        Opens in window (does not save file)
        :param indices: The indices to plot
        :return: void
        '''
        if not indices == [-1]:
            fail = False
            for i in range(len(indices)):
                if indices[i] >= len(self.masses):
                    fail = True
                    break
            if fail:
                print("fun 'plot' indices must be between 0 and len(masses - 1)")
            else:
                for i in range(len(indices)):
                    plt.plot(self.pos_hist(indices[i], 0), self.pos_hist(indices[i], 1))
                plt.show()
        else:
            for i in range(len(self.masses)):
                plt.plot(self.pos_hist(i, 0), self.pos_hist(i, 1))
            plt.show()

    def indices_to_plot(self, index, all):
        '''
        Ancillary function. Returns list of mass indices given a single galaxy index
        :param index: Index of galaxy
        :param boolean all: True - return all indices, False - return indices with to galaxy index 'index'
        :return: Array of indices of masses with galaxy index 'index'
        '''
        indices_to_plot = []
        for i in range(s.no_masses):
            if i != index:
                if all:
                    indices_to_plot.append(i)
                else:
                    if s.mass_indices[i] == index:
                        indices_to_plot.append(i)
        return indices_to_plot

    def crop(self, crop):
        '''
        Ancillary function.
        :param boolean crop: Whether or not to crop an image
        :return: '' if crop == True, 'c' if crop == False
        '''
        r = ''
        if not crop:
            r = 'c'
        return r

    def plot_distances(self, index, times, all, crop):
        '''
        Plots no. of masses within a given distance from mass with index 'index' against this distance at all times 'time'
        :param index: Index of central mass to test
        :param times: Times to make overlaying plots
        :param boolean all: True - plot for all indices, False - plot for indices with to galaxy index 'index'
        :param boolean crop: True - crop image (to pre-determined suitable size), False - do not
        :return: void
        '''
        arr = [[] for _ in range(len(times))]
        indices_to_plot = self.indices_to_plot(index, all)
        for i in indices_to_plot:
            for t in range(len(times)):
                arr[t].append(self.dist(index, i, times[t]))

        no = np.linspace(1, len(indices_to_plot), len(indices_to_plot))
        for t in range(len(times)):
            arr[t].sort(reverse=False)
            plt.plot(arr[t], no, label='t = ' + str(times[t]))
        plt.xlabel('Distance from galactic centre')
        plt.ylabel('Number of masses within this distance')
        if crop:
            plt.xlim(0, 25)
        if len(times) != 1:
            plt.legend()
        plt.savefig('f_' + str(self.crop(crop)) + str(self.timestamp) + '_number_' + str(index) + '.png')
        # clear plot
        plt.clf()

    # Plots density against distance from mass with index 'index' at all times in 'times'
    # 'resolution' is the average number of masses in each bin
    def plot_density(self, index, times, resolution, all, crop):
        '''
        Plots number density against distance from mass at various times
        :param index: Central mass we are considering
        :param times: Times to plot the number density at
        :param resolution: Average number of masses in each bin
        :param boolean all: True - plot for all indices, False - plot for indices with to galaxy index 'index'
        :param boolean crop: True - crop image (to pre-determined suitable size), False - do not
        :return: void
        '''
        indices_to_plot = self.indices_to_plot(index, all)

        dist_divisions = []
        no_bins = 0

        for t in range(len(times)):
            # obtain all distances to central mass
            distances = []
            for i in indices_to_plot:
                distances.append(self.dist(index, i, times[t]))
            max_dist = np.max(distances)

            # divide into no_bins lots of equal bins (ONLY DONE ONCE)
            if t == 0:
                no_bins = int(len(indices_to_plot) / resolution)
                dist_divisions = []
                for i in range(no_bins):
                    dist_divisions.append(i * max_dist / (no_bins - 1))

            # count number of masses in each bin
            number = []
            for i in range(no_bins-1):
                number.append(len(list(x for x in distances if dist_divisions[i] <= x <= dist_divisions[i+1])))

            # plot the central value of each bin against the number in that bin
            central = []
            for i in range(no_bins-1):
                central.append((dist_divisions[i] + dist_divisions[i+1]) / 2)
            plt.plot(central, number, label='t = '+str(times[t]))
        if crop:
            plt.xlim(0, 20)
        plt.xlabel('Distance from galactic centre')
        plt.ylabel('Density of stars')
        if len(times) != 1:
            plt.legend()
        plt.savefig('f_' + str(self.crop(crop)) + str(self.timestamp) + '_density_' + str(index) + '.png')

        # clear plot
        plt.clf()

    def plot_distance_vs_change(self, index, t, all, crop):
        '''
        Plots a scatter of all initial radial distances from the centre of a galaxy against their change in radial distance
        :param index: Central mass we are considering
        :param t: Time step to make plot at
        :param boolean all: True - plot for all indices, False - plot for indices with to galaxy index 'index'
        :param boolean crop: True - crop image (to pre-determined suitable size), False - do not
        :return: void
        '''
        # work out which indices to plot depending on whether 'all' is True
        indices_to_plot = self.indices_to_plot(index, all)

        #  Get data
        distance_init = []
        distance_change = []
        for i in indices_to_plot:
            distance_init.append(self.dist(index, i, 0))
            distance_change.append(self.dist(index, i, t) - self.dist(index, i, 0))

        # Create plot
        plt.plot(distance_init, distance_change, 'k,', markersize=0.05)
        if crop:
            plt.xlim(0, 10)
            plt.ylim(-4, 4)
        plt.xlabel('Initial distance from galactic centre')
        plt.ylabel('Change in distance from galactic centre')
        plt.savefig('f_' + str(self.crop(crop)) + str(self.timestamp) + '_dist_vs_change_' + str(index) + '.png')

        # clear plot
        plt.clf()

    # WARNING: Currently only works in 2D
    def plot_distance_vs_angle(self, index, t, all, crop):
        '''
        Plots a scatter of all radial distances from the centre of a galaxy against their angle
        :param index: Central mass we are considering
        :param t: Time step to make plot at
        :param boolean all: True - plot for all indices, False - plot for indices with to galaxy index 'index'
        :param boolean crop: True - crop image (to pre-determined suitable size), False - do not
        :return: void
        '''
        # work out which indices to plot depending on whether 'all' is True
        indices_to_plot = self.indices_to_plot(index, all)

        distance = []
        angles = []
        for i in indices_to_plot:
            distance.append(self.dist(index, i, t))
            angles.append(self.angle(index, i, t))
        plt.plot(angles, distance, 'k,', markersize=0.05)
        if crop:
            plt.ylim(0, 20)
        plt.ylabel("Distance from galactic centre")
        plt.xlabel("Angle")
        plt.savefig('f_' + str(self.crop(crop)) + str(self.timestamp) + '_dist_vs_angle_' + str(index) + '.png')

        # clear plot
        plt.clf()

    def plot_within_radius(self, index, radii, all):
        '''
        Plots number of stars within given radii as time progresses
        :param index: Central mass we are considering
        :param radii: Array of radii to consider (appear in legend)
        :param boolean all: True - plot for all indices, False - plot for indices with to galaxy index 'index'
        :return: void
        '''
        # work out which indices to plot depending on whether 'all' is True
        indices_to_plot = self.indices_to_plot(index, all)

        data_to_plot = []
        for t in range(len(self.solution)):
            number = [0 for _ in range(len(radii))]
            for i in indices_to_plot:
                for n in range(len(number)):
                    if self.dist(index, i, t) < radii[n]:
                        number[n] += 1
            data_to_plot.append(number)
            np.transpose(data_to_plot)

        ts = np.linspace(0, self.max_time, self.no_steps)
        for i in range(len(radii)):
            plt.plot(ts, np.transpose(data_to_plot)[i], label='r = '+str(radii[i]))
        plt.legend()
        plt.ylabel("Number of masses at a distance less than r")
        plt.xlabel("Time")
        plt.savefig('f_' + str(self.timestamp) + '_no_within_r_' + str(index) + '.png')

        # clear plot
        plt.clf()

    def captured(self, i, j, t):
        '''
        Returns index of all 'captured' from mass with index i by mass with index j
        :param i: First mass index
        :param j: Second mass index
        :param t: Time step being considered
        :return: Index of masses closer to mass i than mass j at time 0, but closer to mass j than mass i at time t
        '''
        r = []
        for k in range(self.no_masses):
            if k != i and k != j:
                if self.dist(i, k, 0) < self.dist(j, k, 0) and self.dist(i, k, t) > self.dist(j, k, t):
                    r.append(k)
        return r

    #  CORE NUMERICAL CALCULATION FUNCTIONS

    def f_grav(self, inp, t, ms):
        '''
        Input function for 'odeint', takes all r_i and dr_i/dt and gives back derivatives
        :param inp: Input vector of the form: [r_0, r_1, ..., r_n, dr_0/dt, dr_1/dt, ..., dr_n/dt]
        :param t: Current time in simulation
        :param ms: Suitably shaped np array of masses (for direct operation on other arrays)
        :return: The time derivative of the input vector
        '''
        no_masses = 2002  # Currently number of light masses must be multiple of 2000 (see report)
        dydt = np.zeros(no_masses*self.d*2)

        # positions --> velocities
        dydt[:no_masses*self.d] = inp[no_masses*self.d:]

        # velocities --> accelerations
        # get displacements in right shape
        cols = inp[: self.d * no_masses].reshape(no_masses, self.d)
        cols_matrix = np.repeat(cols[np.newaxis,...], self.no_heavies, axis=0)
        rows = inp[: self.d * self.no_heavies].reshape(self.no_heavies, self.d)
        rows_matrix = np.reshape(np.repeat(rows[np.newaxis, ...], no_masses, axis=1), (self.no_heavies, no_masses, self.d)) #TODO CHECK
        diff = cols_matrix - rows_matrix
        # get (absolute value of displacements)^3 in right shape
        absr = np.sqrt(np.sum(diff ** 2, axis=2))
        absr_cubed = absr ** 3
        absr_cubed_shaped_1 = np.repeat(absr_cubed[np.newaxis, ...], self.d, axis=2)
        absr_cubed_shaped_2 = np.reshape(absr_cubed_shaped_1, (self.no_heavies, no_masses, self.d))
        absr_cubed_shaped_3 = np.where(absr_cubed_shaped_2 == 0, -1, absr_cubed_shaped_2) # just gets rid of zero absolute and replaces them with 1
        # put it all together and sum over masses
        unsummed_answer = np.where(absr_cubed_shaped_3 < (1**3), -self.G * ms * diff/(1**3), -self.G * ms * diff / absr_cubed_shaped_3)
        summed_answer = np.sum(unsummed_answer, axis=0)
        dydt[no_masses*self.d:] = np.reshape(summed_answer, no_masses*self.d)

        print(t)

        return dydt

    def simulate(self):
        '''
        Core simulation routine: fills in 'self.solution' with the integrated solution by calling 'odeint'
        Note that 'odeint' may be called several times due to memory restrictions
        :return: indicative run times of two parts of the code below
        '''

        # move initial conditions into a vector
        s1 = time.time()
        no_repeats = int((self.no_masses-self.no_heavies)/2000) # Currently number of light masses must be multiple of
        init = []                                               # 2000(see report)
        for j in range(no_repeats):
            tmp = []
            for i in range(self.no_heavies):
                for k in range(self.d):
                    tmp.append(self.masses[i].x_init[k])
            for i in range(2000):
                for k in range(self.d):
                    tmp.append(self.masses[2 + i + j*2000].x_init[k])
            for i in range(self.no_heavies):
                for k in range(self.d):
                    tmp.append(self.masses[i].v_init[k])
            for i in range(2000):
                for k in range(self.d):
                    tmp.append(self.masses[2 + i + j*2000].v_init[k])
            init.append(np.array(tmp, dtype='float64'))
        e1 = time.time()

        s2 = time.time()
        # shape mass array for direct multiplication in f_grav function
        m_shaped = self.mass_vals[:self.no_heavies]
        m_shaped_1 = np.repeat(m_shaped[np.newaxis, ...], self.d*2002, axis=1)
        m_shaped_2 = np.reshape(m_shaped_1, (self.no_heavies, 2002, self.d))
        # set up everything else and loop through calling 'odeint'
        domain = np.linspace(0, self.max_time, self.no_steps)
        solutions_x = []
        solutions_v = []
        for i in range(no_repeats):
            tmp = np.split(np.array(odeint(self.f_grav, init[i], domain, args=(m_shaped_2,))), 2, axis=1)
            # get rid of repeated heavy masses unless i = 0
            if i != 0:
                to_delete = [i for i in range(self.no_heavies*self.d)]
                tmp[0] = np.delete(tmp[0], to_delete, axis=1)
                tmp[1] = np.delete(tmp[1], to_delete, axis=1)
            solutions_x.append(tmp[0])
            solutions_v.append(tmp[1])
        solutions = solutions_x + solutions_v
        self.solution = np.concatenate(solutions, axis=1)
        e2 = time.time()
        self.runtime = e2-s2

        return e1-s1, e2-s2

    #  FUNCTIONS FOR CALCULATION OF PHYSICAL PARAMETERS

    def closest_approach(self, i1, i2):
        '''
        Calculates data at the closes approach between two masses
        :param i1: First mass index
        :param i2: Second mass index
        :return: Tuple of positions and velocities at closest approach of masses with index i1 and i2
        '''

        x1 = self.pos_hist(i1, 0)
        x2 = self.pos_hist(i2, 1)

        r1 = self.solution[:, i1 * self.d:(i1 + 1) * self.d]
        r2 = self.solution[:, i2 * self.d:(i2 + 1) * self.d]
        v1 = self.solution[:, self.d*(self.no_masses + i1):self.d * (self.no_masses + i1 + 1)]
        v2 = self.solution[:, self.d*(self.no_masses + i2):self.d * (self.no_masses + i2 + 1)]

        mod_r = np.sqrt(np.sum((r1-r2)**2, axis=1))
        mod_v = np.sqrt(np.sum((v1-v2)**2, axis=1))

        min_r = np.min(mod_r)
        max_v = np.max(mod_v) # max_v occurs at min_r

        return min_r, max_v

    def calc_ang_mom(self, t):
        '''
        Calculates total angular momentum
        :param t: Time step to calculate it at
        :return: Angular momentum
        '''
        j = 0
        for i in range(self.no_heavies):
            x = self.pos(i, 0, t)
            y = self.pos(i, 1, t)
            vx = self.vel(i, 0, t)
            vy = self.vel(i, 1, t)
            j += self.mass_vals[i]*(x*vy - y*vx)
        return j

    def calc_energy(self, t):
        '''
        Calculates total energy
        :param t: Time step to calculate it at
        :return: Energy
        '''
        energy = 0
        # Kinetic energy
        for i in range(self.no_heavies):
            vx = self.vel(i, 0, t)
            vy = self.vel(i, 1, t)
            energy += 0.5*self.mass_vals[i]*(vx**2 + vy**2)
        # Potential energy
        for i in range(self.no_heavies-1):
            for j in range(self.no_heavies-1-i):
                rx = self.pos(i, 0, t) - self.pos(i + j + 1, 0, t)
                ry = self.pos(i, 1, t) - self.pos(i + j + 1, 1, t)
                energy += -self.G * self.mass_vals[i] * self.mass_vals[i + j + 1] / np.sqrt(rx**2 + ry**2)
        return energy

    def calc_lin_mom(self, t):
        '''
        Calculates liner momentum as a vector
        :param t: Time step to calculate it at
        :return: Liner momentum
        '''
        px = 0
        py = 0
        for i in range(self.no_heavies):
            px += self.mass_vals[i] * self.pos(i, 0, t)
            py += self.mass_vals[i] * self.pos(i+1, 0, t)
        return [px, py]


class Mass:
    def __init__(self, mass, x_init, v_init, colour, galaxy_index):
        '''

        :param mass: The mass of the object
        :param x_init: The initial position as a vector
        :param v_init: The initial velocity as a vector
        :param colour: The colour (for plotting)
        :param galaxy_index: The galaxy it belongs to (if any) corresponding to the mass index of the central object
        '''
        self.colour = colour
        self.mass = mass
        self.x_init = x_init
        self.v_init = v_init
        self.asteroid = (mass == 0)
        self.galaxy_index = galaxy_index


#  FUNCTIONS TO SET UP VARIOUS INITIAL CONDITIONS
#  Parameters have to be passed in since they are outside the classes. They are described only one at the bottom

G = Simulator.G


#  Create ring of equally spaced test masses with initial conditions for circular motion around m_central
def create_ring(r, n, m_central, r_central, v_central, direction, colour, index):
    #  RIGHT: direction = False
    #  LEFT: direction = True
    d = -1 if direction else 1
    lst = []
    for i in range(n):
        lst.append(Mass(0, [r_central[0] + r * np.cos(2 * np.pi * i / n),
                            r_central[1] + r * np.sin(2 * np.pi * i / n)],
                           [v_central[0] + d * (np.sqrt(G * m_central / r)) * np.sin(2 * np.pi * i / n),
                            v_central[1] - d * (np.sqrt(G * m_central / r)) * np.cos(2 * np.pi * i / n)], colour, index))
    return lst


#  Create model galaxy of rings
def create_ring_galaxy(r_central, v_central, direction, m_central, colour, index):
    l = [Mass(m_central, r_central, v_central, colour, index)]
    for i in range(5):
        for j in range(6 * (i + 2)):
            l.append(create_ring(i + 2, 6 * (i + 2), m_central, r_central, v_central, direction, colour, index)[j])
    return l


#  Create more realistic galaxy with exponential radial distribution
def create_galaxy(r_central, v_central, m_central, direction, radius, no_stars, colour, index):
    #  RIGHT: direction = False
    #  LEFT: direction = True
    d = -1 if direction else 1
    rs = np.random.exponential(radius, (no_stars,)) + radius/2
    thetas = 2 * np.pi * np.random.random((no_stars,))
    lst = [Mass(m_central, r_central, v_central, colour, index)]
    for i in range(no_stars):
        lst.append(Mass(0, [r_central[0] + rs[i] * np.cos(thetas[i]),
                            r_central[1] + rs[i] * np.sin(thetas[i])],
                           [v_central[0] + d * (np.sqrt(G * m_central / rs[i])) * np.sin(thetas[i]),
                            v_central[1] - d * (np.sqrt(G * m_central / rs[i])) * np.cos(thetas[i])], colour, index))
    return lst


#  Write file with data
def write_file(s, runtime, end_time, timesteps, rotation_0, rotation_1, radius_0, radius_1, no_0, no_1):
    file = open("data.txt", "a")
    file.write("DATA FOR SIMULATION @ TIME " + str(s.timestamp) + "\n")
    file.write("\n")
    file.write("Runtime                        : " + str(runtime) + "\n")
    file.write("Initial & final energy         : " + str(s.calc_energy(0)) + ", " + str(s.calc_energy(timesteps-1)) + "\n")
    file.write("Initial & finial angular mom.  : " + str(s.calc_ang_mom(0)) + ", " + str(s.calc_ang_mom(timesteps-1)) + "\n")
    file.write("Mass 0 initial conditions      : [" + str(s.pos(0, 0, 0)) + ", " + str(s.pos(0, 1, 0)) + "], [" +
               str(s.vel(0, 0, 0)) + ", " + str(s.vel(0, 1, 0)) + "], " + str(s.mass_vals[0]) + ", " + str(rotation_0) +
               ", " + str(radius_0) + ", " + str(no_0) + "\n")
    file.write("Mass 1 initial conditions      : [" + str(s.pos(1, 0, 0)) + ", " + str(s.pos(1, 1, 0)) + "], [" +
               str(s.vel(1, 0, 0)) + ", " + str(s.vel(1, 1, 0)) + "], " + str(s.mass_vals[1]) + ", " + str(rotation_1) +
               ", " + str(radius_1) + ", " + str(no_1) + "\n")
    file.write("Simulation time                : " + str(end_time) + "\n")
    file.write("Closest approach               : " + str(s.closest_approach(0, 1)) + "\n")
    file.write("No. stars captured (0 --> 1)   : " + str(len(s.captured(0, 1, timesteps-1))) + "\n")
    file.write("No. stars captured (1 --> 0)   : " + str(len(s.captured(1, 0, timesteps-1))) + "\n\n")
    file.write("\n")
    file.close()


#  RUN SIMULATION AND OUTPUT ALL REQUIRED FIlES!

#  init. conditions (this is the only place the code needs to be changed apart from axis limits
#  if required, in animate_2d function)
rotation_0 = True   # Rotation of galaxy 0,  clockwise = False, anticlockwise = True
rotation_1 = False  # Rotation of galaxy 1
radius_0 = 2        # Radius of galaxy 0 - does not correspond directly, but sets up exponential distribution function
radius_1 = 2        # Radius of galaxy 1 - does not correspond directly, but sets up exponential distribution function
no_0 = 10000        # Number of orbiting stars in galaxy 0
no_1 = 10000        # Number of orbiting stars in galaxy 1
max_time = 250      # Computer time the simulation should run for
steps = 625         # Number of time steps the solution should be saved at
r0 = [-20, 0]       # Initial position of central mass in galaxy 0
r1 = [20, 0]        # Initial position of central mass in galaxy 1
v0 = [0.139194109, 0.075]   # Initial velocity of central mass in galaxy 0
v1 = [-0.139194109, -0.075] # Initial velocity of central mass in galaxy 0
m0 = 1              # Mass central mass in galaxy 0
m1 = 1              # Mass central mass in galaxy 1

#  Example simulation and processing of results
g1 = create_galaxy(r0, v0, m0, rotation_0, radius_0, no_0, 'red', 0)
g2 = create_galaxy(r1, v1, m1, rotation_1, radius_1, no_1, 'black', 1)
s = Simulator(g1+g2, 2, max_time, steps)
x, y = s.simulate()
print()
print(str(y) + " seconds for simulation")
print()
print(s.calc_ang_mom(0))
print(s.calc_ang_mom(steps - 1))
print(s.calc_energy(0))
print(s.calc_energy(steps - 1))

s.plot_trajectory_2d([0, 1])
write_file(s, y, max_time, steps, rotation_0, rotation_1, radius_0, radius_1, no_0, no_1)
s.animate_2d('GALAXY COLLISION')

s.plot_distance_vs_angle(1, steps - 1, False, True)
s.plot_distance_vs_angle(0, steps - 1, False, True)
s.plot_distance_vs_angle(1, steps - 1, False, False)
s.plot_distance_vs_angle(0, steps - 1, False, False)

s.plot_distance_vs_change(0, steps - 1, False, True)
s.plot_distance_vs_change(1, steps - 1, False, True)
s.plot_distance_vs_change(0, steps - 1, False, False)
s.plot_distance_vs_change(1, steps - 1, False, False)

s.plot_distances(0, [0, steps - 1], False, True)
s.plot_distances(1, [0, steps - 1], False, True)
s.plot_distances(0, [0, steps - 1], False, False)
s.plot_distances(1, [0, steps - 1], False, False)

s.plot_density(0, [0, steps - 1], 20, False, True)
s.plot_density(1, [0, steps - 1], 20, False, True)
s.plot_density(0, [0, steps - 1], 20, False, False)
s.plot_density(1, [0, steps - 1], 20, False, False)

s.plot_within_radius(0, [2, 4, 8, 16], False)
s.plot_within_radius(1, [2, 4, 8, 16], False)
