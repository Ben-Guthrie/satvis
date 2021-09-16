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
import math
import numpy as np


def julian_date(year, month, day):
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month


    day = day
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)

    D = math.trunc(30.6001 * (monthp + 1))

    jd = B + C + D + day + 1720994.5

    return jd


def get_sun_dir(initial_jd, time, sat_pos):
    # Convert time from seconds to a fraction of a day
    decimal_day = time / 86400.

    # Calculate the julian date
    jd = initial_jd + decimal_day

    # Compute the geocentric longitude and lattitude of the spacecraft (assuming spherical Earth)
    lon, lat = cartesian_to_spherical(sat_pos)[:2]

    # Compute the Sun direction vector using the PSA model
    sun_dir = -psa(jd, lon, lat)

    return sun_dir

def psa(jd, long, lat):
    """Compute the Sun position using the PSA model."""
    # Calculate the ecliptic coordinates of the Sun from the Julian Day
    decimal_day = jd - (math.floor(jd) + 0.5)
    if decimal_day < 0:
        decimal_day += 1
    hour = decimal_day * 24

    n = jd - 2451545
    omega = 2.1429 - 0.0010394594*n
    l_mean_longitude = 4.8950630 + 0.017202791698*n
    g_mean_anomaly = 6.2400600 + 0.0172019699*n

    l_ecliptic_longitude = (l_mean_longitude
                            + 0.03341607*np.sin(g_mean_anomaly)
                            + 0.00034894*np.sin(2*g_mean_anomaly)
                            - 0.0001134
                            - 0.0000203*np.sin(omega))

    ep_obliquity_ecliptic = 0.4090928 - (6.2140e-9)*n + 0.0000396*np.cos(omega)

    # Convert from ecliptic to celestial coordinates
    ra_right_ascenscion = np.arctan2(np.cos(ep_obliquity_ecliptic)*np.sin(l_ecliptic_longitude),
                                     np.cos(l_ecliptic_longitude))

    if ra_right_ascenscion < 0:
        ra_right_ascenscion += 2*np.pi

    delta_declination = np.arcsin(np.sin(ep_obliquity_ecliptic)*np.sin(l_ecliptic_longitude))

    # Convert from celestial to horizontal coordinates
    gmst = 6.6974243242 + 0.0657098283*n + hour
    lmst = (gmst*15+long)*(np.pi/180)
    omega_hour_angle = lmst - ra_right_ascenscion
    theta_z = np.arccos(np.cos(lat)*np.cos(omega_hour_angle)*np.cos(delta_declination)
                        + np.sin(delta_declination)*np.sin(lat))
    gamma = np.arctan2(-np.sin(omega_hour_angle),
                       np.tan(delta_declination)*np.cos(lat) - np.sin(lat)*np.cos(omega_hour_angle))
    earth_mean_radius = 6371.01 # km
    astronomical_unit = 149597890. # km
    parallax = earth_mean_radius * np.sin(theta_z) / astronomical_unit
    theta_z += parallax

    # Convert gamma and theta_z to a unitary direction vector in the topocentric frame
    azim = gamma
    elev = np.pi/2 - theta_z

    gs = np.array([np.sin(elev), np.cos(elev)*np.sin(azim), np.cos(elev)*np.cos(azim)])

    return gs


def cartesian_to_spherical(pos):
    """Returns the longitude, latitude (in rad) and radial distance at the current position."""
    r = np.linalg.norm(pos)
    lat = np.arcsin(pos[2] / r)
    lon = np.arctan2(pos[1], pos[0])
    return lon, lat, r

if __name__ == "__main__":
    print(julian_date(1985, 2, 17.25)
)
