# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:33:26 2019


Copyright (C) 2008-2020 Christian Friesicke <christian.friesicke@iaf.fraunhofer.de>
Copyright (C) 2020 Philipp Neininger <philipp.neininger@iaf.fraunhofer.de>

 Synopsis:
   ntl_bergquist1972.m -- calculate the S-parameters of an arbitrary
   nonuniform transmission line according to Bergquist's method [1].

 Syntax:
   S = nonuniform_smatrix(Z0, gamma, x, precision)

 Input parameters:
   Z0          profile of characteristic impedance [Ohm] along line
   gamma       propagation coefficient [1/m] along line
   x           x-coordinates of discretized line;
               x increases from port 1 towards port 2.
   precision   truncate K,Q series if K(n)/phi{1,2} < precision, or
                                      Q(n)/psi{1,2} < precision.

   Note that Z0, gamma, and x must be one-dimensional vectors of
   equal length.

 Output parameters:
   S           S-parameter matrix of nonuniform transmission line

 Known bugs / limitations:
 * Numerical computation of completely uniform transmission lines results
   in zero-solution for K,Q series.  This may ultimately lead to division
   by zero errors.

 References:
   [1] A. Bergquist -- "Wave Propagation on Nonuniform Transmission
       Lines (Short Papers)", IEEE Trans. MTT, Vol. 20, No. 8, 1972,
       pp. 557--558.
   [2] S. Uysal -- "Nonuniform Line Microstrip Directional Couplers and
       Filters", Artech House, 1993, pp. 9--12; with corrections to
       Eq. 1.28.
"""

import numpy as np
from numpy import exp, abs
from scipy import integrate
import scipy.constants as const


def ntl_bergquist1972(z0, gamma, x, precision):

    # integrate propagation coefficient gamma.
    # system transformation, step 1
    int_gamma = integrate.cumtrapz(gamma + 0j, -x, initial=0j)

    # differentiate characteristic impedance Z0
    ddx_Z0 = np.append(np.diff(z0) / np.diff(x), 0j)

    p = 0.5 * ddx_Z0 / z0 + 0j
    # f1, f2 were corrected from [2] (should be -p)
    f1 = -p * exp(-2 * int_gamma + 0j)
    f2 = -p * exp(2 * int_gamma + 0j)

    # calculate K-series, phi1, phi2.
    nk = 1
    K = dict()
    K[nk] = integrate.cumtrapz(f1, x, initial=0j)
    phi1 = K[nk][-1]
    phi1_precision = 1
    phi2 = 1 + 0j
    phi2_precision = 1
    curr_precision = 1

    while curr_precision > precision:
        nk += 1
        if nk % 2 == 0:
            K[nk] = integrate.cumtrapz(f2 * K[nk-1], x, initial=0j)
            phi2_precision = abs(K[nk][-1] / phi2)
            phi2 += K[nk][-1]
        else:
            K[nk] = integrate.cumtrapz(f1 * K[nk-1], x, initial=0j)
            phi1_precision = abs(K[nk][-1] / phi1)
            phi1 += K[nk][-1]

        curr_precision = max(phi1_precision, phi2_precision)

    # calculate Q-series, psi1, psi2.
    nq = 1
    Q = dict()
    Q[nq] = integrate.cumtrapz(f2, x, initial=0j)
    psi1 = Q[nq][-1]
    psi1_precision = 1
    psi2 = 1 + 0j
    psi2_precision = 1
    curr_precision = 1

    while curr_precision > precision:
        nq += 1
        if nq % 2 == 0:
            Q[nq] = integrate.cumtrapz(f1 * Q[nq-1], x, initial=0j)
            psi2_precision = abs(Q[nq][-1] / psi2)
            psi2 += Q[nq][-1]
        else:
            Q[nq] = integrate.cumtrapz(f2 * Q[nq-1], x, initial=0j)
            psi1_precision = abs(Q[nq][-1] / psi1)
            psi1 = psi1 + Q[nq][-1]

        curr_precision = max(psi1_precision, psi2_precision)

    s11 = -psi1 / phi2
    s12 = exp(int_gamma[-1]) / phi2
    s21 = s12
    s22 = exp(2*int_gamma[-1]) * phi1 / phi2

    return [s11, s12, s21, s22]


def calc_z0(r_vect, d_vect):
    return 376.73 * 8 * d_vect / (r_vect * np.pi)


#  analytical_solution_entl()
# Calculate analytical solution of exponentially tapered NTL (ENTL).
# Solution can be found by transforming the Riccati-type differential
# equation into a 2nd-order ODE with constant coefficients.
#   L: length
def analytical_solution_entl(Z_ratio, L, freq):
    beta = 2 * np.pi * freq / const.c
    q2 = np.log(Z_ratio) / (2*L)
    K = beta * beta - q2**2
    lam1 = 1j*beta + np.sqrt(-K + 0j)
    lam2 = 1j*beta - np.sqrt(-K + 0j)
    Gamma = -1/q2 * (exp(-lam1*L) - exp(-lam2*L)) \
        / (exp(-lam1*L) / lam1 - exp(-lam2*L) / lam2)
    return Gamma


def benchmark():
    n_samples = 500
    L = 20E-3

    f_vect = np.array([1e9, 10e9, 20e9])
    gamma0 = 1j * 2*np.pi * f_vect / const.c
    l_vect = np.linspace(0, L, n_samples)
    z2 = 150
    z1 = 50
    z0_vect = z1 * exp((l_vect / L) * np.log(z2/z1))

    sp = ntl_bergquist1972(z0_vect, np.repeat(gamma0[0], n_samples), l_vect, 0.000001)
    print(np.log10(abs(sp))*20)

    print(np.log10(abs(analytical_solution_entl(z2/z1, L, f_vect)))*20)


if __name__ == "__main__":
    benchmark()
