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


def compute_cw(initial_state, ref_sat, true_anomaly):
    """
    Calculate the relative state of a chaser spacecraft at specific true
    anomalies, given initial conditions. Requires that the target has a
    circular orbit.

    Parameters
    ----------

    initial_state : array of float
        A length-6 vector containing the initial relative state of the chaser
        relative to the target

    ref_sat : Satellite
        The Satellite object for the reference (chaser) satellite

    true_anomaly : float or array of float
        The true anomalies at which to compute the relative state

    Returns
    -------
    states : array of float
        An N x 6 array containing the x,y,z positions and velocities at each time
    """
    assert(np.isclose(ref_sat.e, 0)), "Must be a circular orbit"
    assert(len(initial_state) == 6)
    initial_state = np.array(initial_state, dtype=np.float)
    positions = initial_state[:3]
    velocities = initial_state[3:]
    omega = np.sqrt(ref_sat.mu / ref_sat.r_p**3)  # angular velocity of reference orbit
    assert (omega > 0), "omega must be greater than 0"
    time = np.array(true_anomaly) / ref_sat.omega   # convert anomaly to time

    x_out = cw_x(positions[0], velocities[0], velocities[1], omega, time)
    y_out = cw_y(positions[0], velocities[0], positions[1], velocities[1],
                 omega, time)
    z_out = cw_z(positions[2], velocities[2], omega, time)

    x_dot = d_cw_x(positions[0], velocities[0], velocities[1], omega, time)
    y_dot = d_cw_y(positions[0], velocities[0], velocities[1],
                   omega, time)
    z_dot = d_cw_z(positions[2], velocities[2], omega, time)

    states = np.stack((x_out, y_out, z_out, x_dot, y_dot, z_dot), axis=-1)
    return states


def cw_x(x_0, x_0_dot, y_0_dot, omega, time):
    """
    Calculate the x position given initial condition.

    Parameters
    ----------
    x_0 : float
        The initial x position

    x_0_dot : float
        The initial x velocity

    y_0_dot : float
        The initial y velocity

    omega : float
        The angular rate of the target's orbit

    time : float or array of float
        The time at which to calculate the location of the chaser

    Returns
    -------
    x_out : float or array of float
        The x position at time(s) t

    """
    return x_0_dot/omega * np.sin(omega*time) -\
        (3*x_0 + 2*y_0_dot/omega) * np.cos(omega*time) +\
        (4*x_0 + 2*y_0_dot/omega)


def cw_y(x_0, x_0_dot, y_0, y_0_dot, omega, time):
    """
    Calculate the y position given initial condition.

    Parameters
    ----------
    x_0 : float
        The initial x position

    x_0_dot : float
        The initial x velocity

    y_0 : float
        The initial y position

    y_0_dot : float
        The initial y velocity

    omega : float
        The angular rate of the target's orbit

    time : float or array of float
        The time at which to calculate the location of the chaser

    Returns
    -------
    y_out : float or array of float
        The y position at time(s) t

    """
    return (6*x_0 + 4*y_0_dot/omega) * np.sin(omega*time) +\
        2*x_0_dot/omega * np.cos(omega*time) -\
        (6*omega*x_0 + 3*y_0_dot) * time +\
        (y_0 - 2*x_0_dot/omega)


def cw_z(z_0, z_0_dot, omega, time):
    """
    Calculate the z position given initial condition.

    Parameters
    ----------
    z_0 : float
        The initial z position

    z_0_dot : float
        The initial z velocity

    omega : float
        The angular rate of the target's orbit

    time : float or array of float
        The time at which to calculate the location of the chaser

    Returns
    -------
    z_out : float or array of float
        The z position at time(s) t
    """
    return z_0 * np.cos(omega*time) +\
        z_0_dot/omega * np.sin(omega*time)


def d_cw_x(x_0, x_0_dot, y_0_dot, omega, time):
    """
    Calculate the x velocity given initial condition.

    Parameters
    ----------
    x_0 : float
        The initial x position

    x_0_dot : float
        The initial x velocity

    y_0_dot : float
        The initial y velocity

    omega : float
        The angular rate of the target's orbit

    time : float or array of float
        The time at which to calculate the location of the chaser

    Returns
    -------
    x_out : float or array of float
        The x velocity at time(s) t

    """
    return x_0_dot * np.cos(omega*time) +\
        (3*omega*x_0 + 2*y_0_dot) * np.sin(omega*time)


def d_cw_y(x_0, x_0_dot, y_0_dot, omega, time):
    """
    Calculate the y velocity given initial conditions.

    Parameters
    ----------
    x_0 : float
        The initial x position

    x_0_dot : float
        The initial x velocity

    y_0 : float
        The initial y position

    y_0_dot : float
        The initial y velocity

    omega : float
        The angular rate of the target's orbit

    time : float or array of float
        The time at which to calculate the location of the chaser

    Returns
    -------
    y_out : float or array of float
        The y velocity at time(s) t

    """
    return (8*omega*x_0 + 4*y_0_dot) * np.cos(omega*time) -\
        2*x_0_dot * np.sin(omega*time) -\
        (6*omega*x_0 + 3*y_0_dot)


def d_cw_z(z_0, z_0_dot, omega, time):
    """
    Calculate the z velocity given initial conditions.

    Parameters
    ----------
    z_0 : float
        The initial z position

    z_0_dot : float
        The initial z velocity

    omega : float
        The angular rate of the target's orbit

    time : float or array of float
        The time at which to calculate the location of the chaser

    Returns
    -------
    z_out : float or array of float
        The z velocity at time(s) t

    """
    return -z_0*omega * np.sin(omega*time) +\
        z_0_dot * np.cos(omega*time)
