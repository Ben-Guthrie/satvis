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
from satvis.orbital_dynamics.keplerian_equations import e_h_p_vectors_from_angles, state_at_anomaly



class ReferenceSatellite(object):
    """
    Class for the reference (target) satellite. The state is defined in terms of
    the true anomaly. The orbit is specified by the eccentricity vector and the
    angular momentum vector.
    """
    def __init__(self, eccentricity, angular_momentum, mu, inclination=0,
                    ascending_node=0, periapsis=0):
        self.e = eccentricity # scalar eccentricity
        self.h = angular_momentum # scalar angular momentum
        self.mu = mu
        self.e_vect, self.h_vect, self.p_unitvect = e_h_p_vectors_from_angles(eccentricity,
            angular_momentum, inclination, ascending_node, periapsis)
        self._calculate_keplerian_elements()
        self._calculate_perifocal_frame()

    anomaly = 0.
    all_anomalies = None
    iteration = 0
    num_iterations = 0
    iterable = False

    def _calculate_keplerian_elements(self):
        self.semimajoraxis = self.h**2 / (self.mu * (1 - self.e**2))
        self.r_p = self.semimajoraxis * (1 - self.e) # radius at perigee
        self.r_a = self.semimajoraxis * (1 + self.e) # radius at apogee
        self.semilatusrectum = self.semimajoraxis * (1 - self.e**2)
        self.period = 2*np.pi * np.sqrt(self.semimajoraxis**3 / self.mu) # time period for one full rotation
        self.omega = 2*np.pi / self.period  # angular velocity of reference orbit

    def _calculate_perifocal_frame(self):
        self.omega_unitvect = self.h_vect / self.h
        self.q_unitvect = np.cross(self.omega_unitvect, self.p_unitvect)

    def get_state(self):
        return state_at_anomaly(self, self.anomaly)

    def set_states(self, anomalies):
        """Setup the satellite for iteration through the calculated anomalies."""
        self.all_anomalies = anomalies
        self.anomaly = anomalies[0]
        self.iterable = True
        self.num_iterations = len(anomalies)

    def set_iter(self, i):
        if i >= self.num_iterations:
            return False
        self.iteration = i
        return True

    def iterate(self):
        if self.iterable:
            self.iteration += 1
            if self.iteration < self.num_iterations:
                self.anomaly = self.all_anomalies[self.iteration]
                return True
            else:
                return False

    def __str__(self):
        str = ("Reference satellite\n" +
               "-------------------\n" +
               "e = {}, h = {}, e_vect = {}, " +
               "period = {}").format(self.e, self.h, self.e_vect, self.period)
        return str
