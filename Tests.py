import numpy as np
from functools import partial
import time


xs = np.array([[1,2,3,4,5],[10,20,30,40,50]])
ys = np.array([], dtype=np.int64).reshape(0,5)
xs = np.vstack([ys, xs])
print(xs)


# def acc_towards(rs, accs, earth):
#     """
#     Tests whether the acceleration is towards the Earth/moon for masses in contact with Earth/moon
#     :param rs: Position vectors centred at Earth/moon
#     :param accs: Acceleration vectors
#     :param earth: True if testing Earth, False if testing moon
#     :return: Array of bools, True if above two conditions satisfied, False if not
#     """
#     contact = gEarth_contact if earth else gMoon_contact
#     r_angles = np.where(contact, [np.arctan(rs[i + 1] / rs[i]) for i in range(0, rs.size, 2)], 0)
#     axs = accs[::2]
#     ays = accs[1::2]
#     a_perp = np.where(contact, axs*np.cos(r_angles) + ays*np.sin(r_angles), 0)
#     return np.logical_and(contact, a_perp < 0)
#
#
# def e1(t, r):
#     """
#
#     :param t: Current time in simulation
#     :param r: Position vector (from CoM) of one of the masses
#     :return:
#     """
#     return np.sqrt(np.sum(r ** 2)) - 1
#
#
# gEarth_contact = [True, False, False]
# gMoon_contact = [False, False, False]
# ACCS = np.array([0.02, 0.001, 0.7, 0.3, 1, 8])
# RS = np.array([0.99999, 0.02, 1.72, 0.84, 0.2, 1.976])
#
# d_e = {f'e_crash_{k}': partial(e1, r=RS[2 * k: 2 * (k + 1)]) for k in range(3)}
# events = []
# for _, fn in d_e.items():
#     events.append(fn)
#     fn.terminal = True
#     fn.direction = -1
#
# rce = acc_towards(RS, ACCS, True)
#
# gEarth_contact = np.where(gEarth_contact,
#                           np.where(np.logical_not(rce),
#                                    [False if events[i](0) > 0 else True for i in range(3)],
#                                    gEarth_contact),
#                           gEarth_contact)
#
# ACCS = np.where(np.repeat(gEarth_contact, 2), np.where(np.repeat(rce, 2), 0, ACCS), ACCS)
#
# #print(ACCS)
# print(gEarth_contact)
# print(ACCS)


# import numpy as np
# from scipy.integrate import solve_ivp
# from functools import partial
# import scipy
# import sys
#
# def f(t, y):
#     r = [0, 0]
#     r[0] = y[1]
#     r[1] = -5*y[0]
#     return r
#
#
# # # Crosses x=0.5
# # def e1(t, y):
# #     return y[0] - 0.5
# # e1.terminal=True
# #
# #
# # # Crosses x=0
# # def e2(t, y):
# #     return y[0]
# # e2.terminal=True
#
# def e(t, y, i):
#     return y[0] - 0.75*(1-i)
#
#
# d = {f'e{k}': partial(e, i=k) for k in range(3)}
# es = []
# for _, fn in d.items():
#     es.append(fn)
#     fn.terminal = True
#
#
# sol = solve_ivp(f, (0, 10), (1, 0), t_eval=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], events=[es[0], es[1]])
# iv = [(sol.y_events[i].flatten(), i, sol.t_events[i][0]) for i in range(len(sol.t_events)) if sol.t_events[i]][0]
#
# s = sol.y
# t = sol.t_events
#
# # ABOVE NOW WORKS BUT GIVES A WARNING...
#
#
# # index = 0
# # val = 0
# # for i in range(len(sol.t_events)):
# #     if sol.t_events[i]:
# #         index = i
# #         val = sol.t_events[i][0]
# #         break


