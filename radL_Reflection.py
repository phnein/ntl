# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:10:10 2019

@author: neininger
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from numpy import sqrt, abs, exp
from scipy.optimize import Bounds, minimize

from ntl_bergquist1972 import ntl_bergquist1972

rmin = 3e-3
rmax = 18e-3
deltah = -0.5e-3
h1 = 1.5e-3
k = 500
n_mesh = 1001
f0 = 26e9
f1 = 40e9
n_freq_samples = 101
f_vect = np.linspace(f0, f1, n_freq_samples)
gamma_vect = 1j * 2*np.pi * f_vect / const.c
r = np.linspace(rmin, rmax, n_mesh)

plt.rcParams["axes.labelsize"] = 'large'
plt.rcParams['legend.title_fontsize'] = 11


def radL_z0(d, r):
    return sqrt(const.mu_0 / const.epsilon_0) * 8 / np.pi * d / r


def get_s11_step_taper(stp1_dh_rel, stp1_r0, stp2_r0):
    stp1_dh_abs = -deltah * stp1_dh_rel
    stp2_dh_abs = -deltah * (1 - stp1_dh_rel)

    d_stp = h1 + deltah + stp1_dh_abs * (r > stp1_r0) + stp2_dh_abs * (r > stp2_r0)
    z_dstp = radL_z0(d_stp, r)

    s11_dstp = list()
    for gamma in gamma_vect:
        s11_dstp.append(20*np.log10(abs(
                ntl_bergquist1972(z_dstp, np.repeat(gamma, n_mesh), r, 0.001)[0])))

    return s11_dstp


def sum_of_sqr_err(x, goal_s11):
    stp1_dh_rel = x[0]
    stp1_r0 = x[1]
    stp2_r0 = x[2]

    s11 = np.array(get_s11_step_taper(stp1_dh_rel, stp1_r0, stp2_r0))
    s11_err = s11[s11 > goal_s11]
    sumsq = np.sum(np.square(s11_err - goal_s11))
    print(f"({x[0]:2.2f}, {x[1]*1e3:2.2f}mm, {x[2]*1e3:2.2f}mm) -- {sumsq:4.2f}")

    return sumsq


def plot_s11(x):
    stp1_dh_rel = x[0]
    stp1_r0 = x[1]
    stp2_r0 = x[2]
    s11 = np.array(get_s11_step_taper(stp1_dh_rel, stp1_r0, stp2_r0))
    plt.plot(f_vect/1e9, s11)
    plt.ylim(-40, 0)
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("S_{11} [dB]")
    plt.grid()
    sum_of_sqr_err(x, -30.)


def optimize_step_taper():
    bounds = Bounds([0, rmin, rmin], [1, rmax, rmax])
    res = minimize(sum_of_sqr_err, [0.678, 4e-3, 6e-3], method='SLSQP',
                   options={'xtol': 1e-3, 'disp': True},
                   args=(-30.),
                   bounds=bounds)
    print(res)


def show_comparison():
    z_dconst = radL_z0(np.repeat(h1+deltah, n_mesh), r)

    z_dlin = radL_z0(np.linspace(h1+deltah, h1, n_mesh), r)

    d_exp = h1 + deltah*exp(-k*(r-rmin))
    z_dexp = radL_z0(d_exp, r)

    stp1_dh_rel = 0.678
    stp1_r0 = 4e-3
    stp2_r0 = 6e-3
    stp1_dh_abs = -deltah * stp1_dh_rel
    stp2_dh_abs = -deltah * (1 - stp1_dh_rel)
    d_stp = h1 + deltah + stp1_dh_abs * (r > stp1_r0) + stp2_dh_abs * (r > stp2_r0)
    z_dstp = radL_z0(d_stp, r)

#    p = plt.plot(r*1e3, 1e3*np.repeat(h1+deltah, n_mesh), label='Constant')
#    plt.vlines([1e3*rmin, rmax*1e3], 0, 1e3*(deltah+h1), color=p[0].get_color())
#    plt.hlines(0, rmin*1e3, rmax*1e3, color=p[0].get_color())
#    p = plt.plot(r*1e3, 1e3*np.linspace(h1+deltah, h1, n_mesh), label='Linear')
#    plt.vlines(rmax*1e3,  1e3*h1, 1e3*(deltah+h1), color=p[0].get_color())
#    plt.plot(r*1e3, 1e3*d_stp, label='Steps')
#    plt.plot(r*1e3, 1e3*d_exp, label='Exp.')
#    plt.xlabel('Radius in mm')
#    plt.ylabel('Height in mm')
#    plt.ylim(-1, 2)
#    plt.legend(title='Profiles:', ncol=2, loc='lower right')
#    plt.tight_layout()
#    plt.savefig("Profiles_HeightTaperMethods_srcPlot.svg")
#    plt.show()

    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_subplot(111)
    ax.plot(r*1e3, z_dconst, label='Constant')
    ax.plot(r*1e3, z_dlin, label='Linear')
    ax.plot(r*1e3, z_dstp, label='Steps')
    ax.plot(r*1e3, z_dexp, label='Exp.')
    ax.set_xlabel('Radius [mm]')
    ax.set_ylabel('Z [Ohm]')
    ax.set_xlim(3, 18)
    ax.set_ylim(50)
    ax.grid(ls='--')
    ax.legend(title='Height $h(r)$:', fontsize=11)
    fig.tight_layout()
    plt.savefig("Z_vs_HeightTaperMethods.pdf")

#    plt.plot(r*1e3, z_dconst, label='Constant')
#    plt.xlabel('Radius in mm')
#    plt.ylabel('Z in Ohm')
#    plt.grid(ls='--')
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig("Z_vs_ConstHeight.pdf")
#    plt.show()

    s11_dconst = list()
    s11_dlin = list()
    s11_dstp = list()
    s11_dexp = list()
    for gamma in gamma_vect:
        s11_dconst.append(20*np.log10(abs(
                ntl_bergquist1972(z_dconst, np.repeat(gamma, n_mesh), r, 0.001)[0])))
        s11_dlin.append(20*np.log10(abs(
                ntl_bergquist1972(z_dlin, np.repeat(gamma, n_mesh), r, 0.001)[0])))
        s11_dstp.append(20*np.log10(abs(
                ntl_bergquist1972(z_dstp, np.repeat(gamma, n_mesh), r, 0.001)[0])))
        s11_dexp.append(20*np.log10(abs(
                ntl_bergquist1972(z_dexp, np.repeat(gamma, n_mesh), r, 0.001)[0])))

    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_subplot(111)
    ax.plot(f_vect/1e9, s11_dconst, label='Constant')
    ax.plot(f_vect/1e9, s11_dlin, label='Linear')
    ax.plot(f_vect/1e9, s11_dstp, label='Steps')
    ax.plot(f_vect/1e9, s11_dexp, label='Exponential')
    ax.set_ylim(-50, -10)
    ax.set_yticks([-50, -40, -30, -20, -10])
    ax.set_xlim(26, 40)
    ax.set_xticks(np.linspace(26, 40, 8))
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('$S_{11}$ [dB]')
    ax.grid(ls='--')
    plt.tight_layout()
    plt.savefig("S11_vs_HeightTaperMethods.pdf")


if __name__ == "__main__":
    print("Showing Comparison")
    show_comparison()
