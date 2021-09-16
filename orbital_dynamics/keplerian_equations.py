"""
    satvis: On-orbit visualisation of satellite proximity operations
    Copyright (C) 2021  Ben Guthrie

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from numpy import cos, sin


def e_h_p_vectors_from_angles(eccentricity, angular_momentum, inclination=0,
                    ascending_node=0, periapsis=0):
    """
    Get the eccentricity and angular momentum vectors to initialise the
    satellite object, from the scalar eccentricity/angular momentum and the
    orbit angles.

    Parameters
    ----------
    eccentricity :
    """
    # Abbreviate parameters
    W, w, i = ascending_node, periapsis, inclination

    unitvect_p = np.array([cos(W)*cos(w) - cos(i)*sin(W)*sin(w),
                           cos(w)*sin(W) + cos(W)*cos(i)*sin(w),
                           sin(i)*sin(w)])
    unitvect_w = np.array([sin(i)*sin(W),
                           -sin(i)*cos(W),
                           cos(i)])
    e_vector = eccentricity * unitvect_p
    h_vector = angular_momentum * unitvect_w

    return e_vector, h_vector, unitvect_p


def e_h_from_initial_state(state, mu):
    r_vector = state[:3]
    v_vector = state[3:]
    h_vector = np.cross(r_vector, v_vector)
    e_vector = (np.cross(v_vector, h_vector)/mu -
                r_vector / np.linalg.norm(r_vector))
    # determine the initial anomaly
    theta = np.arccos(np.dot(r_vector, e_vector) /
                      (np.linalg.norm(r_vector) * np.linalg.norm(e_vector)))
    # Resolve the sign of theta
    # determine whether the component of r tangential to the plane of (e,h) is +ve or -ve
    tangent = np.cross(h_vector, e_vector)
    comp = np.dot(r_vector, tangent)
    if comp < 0:
        theta = -theta
    return e_vector, h_vector, theta


def state_at_anomaly(sat, anomaly):
    anomaly = np.atleast_1d(anomaly)
    # Using perifocal frame
    p = sat.p_unitvect[np.newaxis, :]
    q = sat.q_unitvect[np.newaxis, :]
    # Calculate position
    r_abs = sat.semilatusrectum / (1 + sat.e * cos(anomaly))
    a = np.multiply(r_abs, cos(anomaly))[:, np.newaxis]
    b = np.multiply(r_abs, sin(anomaly))[:, np.newaxis]
    r_vector = np.multiply(a, p) + np.multiply(b, q)
    # Calculate velocity
    anomaly = anomaly[:, np.newaxis]
    v_vector = sat.mu / sat.h * (np.multiply(-sin(anomaly), p) +
                                              np.multiply((sat.e+cos(anomaly)), q))
    return np.squeeze(np.concatenate((r_vector, v_vector), axis=1))


def anomaly_from_time(sat, time):
    # if the orbit is circular, the relationship is linear
    if np.isclose(sat.e, 0):
        omega = sat.h / sat.r_p**2
        return time * omega
    else:
        # Solve Kepler's equation
        M = 2*np.pi / sat.period * time

        def fx(x_n):
            return x_n - sat.e*np.sin(x_n) - M

        def dfdx(x_n):
            return 1 - sat.e*np.cos(x_n)
        x_0 = M
        eccentric_anomaly = newton(fx, dfdx, x_0)
        true_anomaly = 2*np.arctan(np.sqrt((1+sat.e) / (1-sat.e)) *
                                   np.tan(eccentric_anomaly/2))
        # Convert from [-pi, pi] to [0, 2pi]
        true_anomaly = np.atleast_1d(true_anomaly)
        true_anomaly[true_anomaly < 0] += 2*np.pi
        # Add multiples of 2pi
        true_anomaly += time // sat.period * (2*np.pi)
        # Correct for if time is an exact multiple of period
        true_anomaly = np.where((time % sat.period) == 0., time // sat.period * (2*np.pi), true_anomaly)
        return np.squeeze(true_anomaly)


def time_from_anomaly(sat, anomaly):
    if np.isclose(sat.e, 0):
        omega = sat.h / sat.r_p**2
        return anomaly / omega
    else:
        # Solved direct time integration
        h = sat.h
        mu = sat.mu
        e = sat.e

        time = (h**3 / (mu**2 * (1-e**2)**(1.5)) *
                (2*np.arctan(np.sqrt((1-e) / (1+e)) * np.tan(anomaly/2.)) -
                 (e*np.sqrt(1-e**2)*np.sin(anomaly)) / (1 + e*np.cos(anomaly))))
        # Convert negative times to positive
        time = np.atleast_1d(time)
        time[time < 0] += sat.period
        # Add multiples of period
        time += (anomaly // (2*np.pi) * sat.period *
                 ((anomaly % (2*np.pi)) > 0.))
        # Correct for if anomaly is an exact multiple of 2pi
        time = np.where((anomaly % (2*np.pi)) == 0., anomaly // (2*np.pi) * sat.period, time)
        return time


def get_change_in_anomaly(anomaly):
    df = anomaly[1:] - anomaly[:-1]
    df[np.where(df < 0)] = 2*np.pi + df[np.where(df < 0)]

    return df


def newton(fx, dfdx, x_0, tol=1e-10):
    Nsteps = 100
    x_norm_change = 1
    step = 0
    x = x_0
    initial_magnitude = np.linalg.norm(x_0)
    while abs(x_norm_change) > tol and step < Nsteps:
        step = step+1
        x_old = x.copy()
        x = x - fx(x)/dfdx(x)  # Newton's method
        x_norm_change = np.linalg.norm(x - x_old) / initial_magnitude
    if step >= Nsteps:
        raise Exception("Newton's method has not converged after {} steps".format(Nsteps))
    return x
