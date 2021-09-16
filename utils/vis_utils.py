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
import os
import numpy as np
import bpy
import mathutils
import math
from satvis.utils.blender_utils import make_sat, make_sun, \
                                    make_camera, make_torch, setup_depth, rotate_sat, \
                                    rotate_earth, save_render, get_data, CAMERA_VIEW_DIRECTION, \
                                    get_sat_path, move_sun, move_chaser
from satvis.orbital_dynamics.sun_position import julian_date, get_sun_dir
from satvis.orbital_dynamics.satellite import ReferenceSatellite
from satvis.orbital_dynamics.clohessy_wiltshire import compute_cw

def setup_scene(configs):
    bpy.context.scene.render.resolution_x = configs["height"]
    bpy.context.scene.render.resolution_y = configs["height"]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.world.horizon_color = (0, 0, 0)
    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = 0.02

def setup_params_partial(configs, params, depth_path):
    # Setup parameters
    initial_pos = np.array(configs["initial_pos"], dtype=np.float32) * configs["distance_scale"]
    configs["cam_pos"] = np.array(configs["cam_pos"], dtype=np.float32) * configs["distance_scale"]
    configs["max_depth"] = configs["max_depth"] * configs["distance_scale"]
    configs["attitude"] = [math.radians(float(i)) for i in configs["attitude"]]

    # If ang_vel is not a list of euler angles, get the rotation from axis angle
    if not isinstance(configs["ang_vel"], (list, np.ndarray)):
        axis = configs["axis"]
        axis /= np.linalg.norm(axis)
        # Rotate axis to line up with the camera view direction
        rotation_diff = mathutils.Vector(CAMERA_VIEW_DIRECTION).rotation_difference(mathutils.Vector(configs["cam_dir"]))
        axis = mathutils.Vector(axis)
        axis.rotate(rotation_diff)
        # Work out the rotation per step as a quaternion
        angle = math.radians(configs["ang_vel"])
        rotation_step = mathutils.Quaternion(axis, angle)
    else:
        ang_vel = [math.radians(float(i)) for i in configs["ang_vel"]]
        rotation_step = mathutils.Euler(ang_vel, 'XYZ').to_quaternion()
    # Convert angular velocity into quaternion rotation per step
    axis, angle = rotation_step.to_axis_angle()
    rotation_step = mathutils.Quaternion(axis, angle*configs["frame_skip"]/configs["fps"])

    if params.sun_behind:
        configs["sun_dir"] = 0

    # If sun_dir is an angle, rotate this much from the view direction, around the world z axis
    if not isinstance(configs["sun_dir"], (list, np.ndarray)):
        angle = configs["sun_dir"] * np.pi / 180.
        Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]]) # rotation matrix about z
        sun_dir = Rz @ configs["cam_dir"]
        configs["sun_dir"] = sun_dir

    # Make objects
    earth = bpy.data.objects['Earth']
    earth.location = initial_pos
    earth.scale = np.array([1,1,1]) * 637100 * configs['distance_scale']
    #earth.transform_apply(scale=True)
    sat_initial_attitude = mathutils.Euler(configs["attitude"], 'XYZ').to_quaternion()
    sat_path = get_sat_path(configs["satObjDir"], configs["target"])
    sat = make_sat(sat_path, sat_initial_attitude, configs["distance_scale"])
    sun = make_sun(configs["sun_dir"], -10000000*configs["distance_scale"]*np.array(configs["sun_dir"]))
    cam = make_camera(configs["cam_pos"], configs["cam_dir"], configs["distance_scale"], configs["fov"])
    torch = make_torch(configs["cam_pos"], configs["cam_dir"], configs["distance_scale"], use_torch=not params.no_torch)
    setup_depth(depth_path)

    # Set clip distance
    cam.data.clip_end = 1.5 * np.linalg.norm(initial_pos)
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            for s in a.spaces:
                if s.type == 'VIEW_3D':
                    s.clip_end = 1.5 * np.linalg.norm(initial_pos)

    return rotation_step, earth, sat, sun, cam, torch

def setup_params_full(configs, params, depth_path):
    # Setup random orbit
    inclination, ascending_node, periapsis = np.random.rand(3) * np.pi
    mu = float(configs["mu"])
    angular_momentum = np.sqrt(mu * configs["radius"])
    ref_sat = ReferenceSatellite(0, angular_momentum, mu, inclination, ascending_node, periapsis)
    print(ref_sat)

    # Start at a random point in orbit
    num_steps_in_orbit = ref_sat.period * configs["fps"]
    start_iter = np.random.randint(num_steps_in_orbit)
    times = np.arange(configs["duration"]*configs["fps"]+1) / configs["fps"] + start_iter
    anomalies = times / num_steps_in_orbit * 2*np.pi
    anomalies = anomalies % (2*np.pi)
    ref_sat.set_states(anomalies)

    # Convert date and time to Julian date
    date = configs["date"].split('/')
    assert(len(date) == 3 and len(date[2]) == 4), 'Date should be in the form dd/mm/yyyy'
    year, month, day = [float(i) for i in date]
    day += float(configs["time"]) / 24.
    jd = julian_date(year, month, day)

    # Get initial position of Earth, Sun and chaser
    initial_pos = np.array(configs["radius"] * configs["distance_scale"]) * np.array([-1, 0, 0])
    configs["max_depth"] = configs["max_depth"] * configs["distance_scale"]
    configs["attitude"] = [math.radians(float(i)) for i in configs["attitude"]]
    sun_dir = get_sun_dir(jd, 0, ref_sat.get_state()[:3])

    # Get states of target and chaser
    # If ang_vel is not a list of euler angles, get the rotation from axis angle
    if not isinstance(configs["ang_vel"], (list, np.ndarray)):
        axis = mathutils.Vector(configs["axis"])
        axis.normalize()
        # Work out the rotation per step as a quaternion
        angle = math.radians(configs["ang_vel"])
        rotation_step = mathutils.Quaternion(axis, angle)
    else:
        ang_vel = [math.radians(float(i)) for i in configs["ang_vel"]]
        rotation_step = mathutils.Euler(ang_vel, 'XYZ').to_quaternion()
    # Convert angular velocity into quaternion rotation per step
    axis, angle = rotation_step.to_axis_angle()
    rotation_step = mathutils.Quaternion(axis, angle*configs["frame_skip"]/configs["fps"])
    # Get chaser states
    chaser_initial_state = np.array([configs["distance"], 0, 0, 0, -2*ref_sat.omega*configs["distance"], 0]) # Initial state for a circular relative orbit
    chaser_pos = chaser_initial_state[:3] * configs["distance_scale"]
    chaser_dir = -chaser_initial_state[:3] / np.linalg.norm(chaser_initial_state[:3])
    chaser_states = compute_cw(chaser_initial_state, ref_sat, anomalies)

    # Get sun direction at each step
    sun_dirs = []
    for i, t in enumerate(times):
        ref_sat.set_iter(i)
        sun_dirs.append(get_sun_dir(jd, times[i], ref_sat.get_state()))

    # Make objects
    earth = bpy.data.objects['Earth']
    earth.location = initial_pos
    earth.scale = np.array([1,1,1]) * 637100 * configs['distance_scale']
    #earth.transform_apply(scale=True)
    sat_initial_attitude = mathutils.Euler(configs["attitude"], 'XYZ').to_quaternion()
    sat_path = get_sat_path(configs["satObjDir"], configs["target"])
    sat = make_sat(sat_path, sat_initial_attitude, configs["distance_scale"])
    sun = make_sun(sun_dir, -10000000*configs["distance_scale"]*np.array(sun_dir))
    cam = make_camera(chaser_pos, chaser_dir, configs["distance_scale"], configs["fov"])
    torch = make_torch(chaser_pos, chaser_dir, configs["distance_scale"], use_torch=not params.no_torch)
    setup_depth(depth_path)

    # Set clip distance
    cam.data.clip_end = 1.5 * np.linalg.norm(initial_pos)
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            for s in a.spaces:
                if s.type == 'VIEW_3D':
                    s.clip_end = 1.5 * np.linalg.norm(initial_pos)

    return rotation_step, earth, sat, sun, cam, torch, ref_sat.period, chaser_states, sun_dirs

def animate(configs, params, save_path, data_path, rotation_step, earth, sat, sun, cam, torch, period, chaser_states=None, sun_dirs=None):
    frame_end = configs["duration"] * configs["fps"]
    bpy.context.scene.frame_end = configs["duration"] * configs["fps"]
    earth_rot_angle = configs["frame_skip"] / configs["fps"] / period * 2 * np.pi

    # Rotate satellite by constant angular velocity
    for i in range(frame_end//configs["frame_skip"]+1):
        rotate_sat(sat, i*configs["frame_skip"], rotation_step)
        rotate_earth(earth, i*configs["frame_skip"], earth_rot_angle)
        # If doing a full simulation, move chaser and sun accordingly to the dynamics
        if chaser_states is not None:
            iter = i * configs["frame_skip"]
            move_sun(sun, i, sun_dirs[iter], -1E8*configs["distance_scale"]*sun_dirs[iter])
            move_chaser(cam, torch, i, chaser_states[iter,:3]*configs["distance_scale"], -chaser_states[iter,:3]/np.linalg.norm(chaser_states[iter,:3]))

    # Save animation
    data = []
    for i in range(frame_end//configs["frame_skip"]+1):
        bpy.context.scene.frame_set(i*configs["frame_skip"])
        save_render(i, save_path)
        data.append(get_data(sat, cam, i/configs["fps"]*configs["frame_skip"], configs["distance_scale"]))

    # Save data to file
    np.savetxt(os.path.join(data_path, "data.txt"), data, header="pos[3], target q[4], chaser q[4], time")
