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

CAMERA_VIEW_DIRECTION = [0, 0, -1]

def get_sat_path(sat_dir, sat_name):
    for filename in os.listdir(sat_dir):
        if os.path.isfile(os.path.join(sat_dir, filename)):
            name, ext = os.path.splitext(filename)
            if name == sat_name and ext != ".mtl":
                sat_path = os.path.join(sat_dir, filename)
                return sat_path
    raise ValueError("Could not find the target model file!")

def make_sat(objPath, initialRot, distanceScale):
    extension = os.path.splitext(objPath)[1]
    if extension == ".obj":
        bpy.ops.import_scene.obj(filepath=objPath)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=objPath)
    elif extension == ".x3d":
        bpy.ops.import_scene.x3d(filepath=objPath)
    elif extension == ".3ds":
        bpy.ops.import_scene.autodesk_3ds(filepath=objPath)
    else:
        raise IOError("Cannot load " + extension + " files. Please use obj, fbx, x3d or 3ds.")
    name = os.path.splitext(os.path.basename(objPath))[0]
    sat = bpy.data.objects[name]

    # Select the newly created object
    #bpy.context.view_layer.objects.active = sat
    sat.select = True

    #bpy.ops.object.modifier_add(type='SUBSURF')
    #bpy.ops.object.shade_smooth()

    # Scale to approx. 10m size
    sat.scale = np.array([1,1,1]) * 10 * distanceScale

    # Rotate
    sat.rotation_mode = 'QUATERNION'
    sat.rotation_quaternion = initialRot
    return sat

def make_sun(dir, pos):
    #bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0))
    bpy.ops.object.lamp_add(type='SUN', radius=1, location=pos)
    sun = bpy.data.objects['Sun']

    # Rotate
    angle = np.arccos(np.dot(dir, CAMERA_VIEW_DIRECTION))
    axis = np.cross(CAMERA_VIEW_DIRECTION, dir)
    sun.rotation_mode = "AXIS_ANGLE"
    sun.rotation_axis_angle = np.concatenate([[angle], axis])

    sun.data.energy = 2
    sun.data.shadow_method = 'RAY_SHADOW'

    return sun

def make_camera(pos, dir, distanceScale, fov):
    bpy.ops.object.camera_add(location=pos)
    cam = bpy.data.objects['Camera']

    # Rotate
    angle = np.arccos(np.dot(dir, CAMERA_VIEW_DIRECTION))
    axis = np.cross(CAMERA_VIEW_DIRECTION, dir)
    cam.rotation_mode = "AXIS_ANGLE"
    cam.rotation_axis_angle = np.concatenate([[angle], axis])

    # Change FOV
    cam.data.lens_unit = 'FOV'
    cam.data.angle = fov / 180. * np.pi
    cam.data.clip_start = 5 * distanceScale
    cam.data.clip_end = 250000 * distanceScale

    bpy.context.scene.camera = cam
    return cam

def make_torch(pos, dir, distanceScale, use_torch=True):
    bpy.ops.object.lamp_add(type='SPOT', radius=1, location=pos)
    torch = bpy.data.objects['Spot']

    # Rotate
    angle = np.arccos(np.dot(dir, CAMERA_VIEW_DIRECTION))
    axis = np.cross(CAMERA_VIEW_DIRECTION, dir)
    torch.rotation_mode = "AXIS_ANGLE"
    torch.rotation_axis_angle = np.concatenate([[angle], axis])

    if not use_torch:
        torch.data.energy = 0
    else:
        torch.data.energy = 1.
    torch.data.distance = 30. * distanceScale

    return torch

def setup_depth(depth_filepath):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    rl = tree.nodes["Render Layers"]
    depthOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    depthOutput.format.file_format = "OPEN_EXR"
    depthOutput.inputs.remove(depthOutput.inputs[0])
    depthOutput.file_slots.new("00")
    depthOutput.base_path = depth_filepath
    #fileSlot = depthOutput.file_slots[0]
    #fileSlot.format.file_format = "OPEN_EXR"
    #fileSlot.use_node_format = False

    links.new(rl.outputs[2], depthOutput.inputs[0])

    return depthOutput

def rotate_sat(sat, i, rotation):
    bpy.context.scene.frame_set(i)
    if i > 0:
        sat.delta_rotation_quaternion = sat.delta_rotation_quaternion * rotation

    sat.keyframe_insert(data_path='delta_rotation_quaternion')

    setKFInterpolation(sat, 'LINEAR')

def rotate_earth(earth, i, rotation_angle):
    bpy.context.scene.frame_set(i)
    if i > 0:
        earth.delta_rotation_euler[2] = earth.delta_rotation_euler[2] - rotation_angle

    earth.keyframe_insert(data_path='delta_rotation_euler')

    setKFInterpolation(earth, 'LINEAR')

def move_sun(sun, i, dir, pos):
    bpy.context.scene.frame_set(i)
    angle = np.arccos(np.dot(dir, CAMERA_VIEW_DIRECTION))
    axis = np.cross(CAMERA_VIEW_DIRECTION, dir)
    sun.rotation_axis_angle = np.concatenate([[angle], axis])
    sun.location = pos
    sun.keyframe_insert(data_path='rotation_axis_angle')
    sun.keyframe_insert(data_path='location')


def move_chaser(cam, torch, i, pos, dir):
    bpy.context.scene.frame_set(i)
    angle = np.arccos(np.dot(dir, CAMERA_VIEW_DIRECTION))
    axis = np.cross(CAMERA_VIEW_DIRECTION, dir)
    for obj in [cam, torch]:
        obj.location = pos
        obj.rotation_axis_angle = np.concatenate([[angle], axis])
        obj.keyframe_insert(data_path='rotation_axis_angle')
        obj.keyframe_insert(data_path='location')

def setKFInterpolation(obj, interpolation_type):
    fcurves = obj.animation_data.action.fcurves
    for fcurve in fcurves:
        kf = fcurve.keyframe_points[-1]
        kf.interpolation = interpolation_type

def save_render(i, img_filepath):
    bpy.context.scene.render.filepath = os.path.join(img_filepath,str(i).zfill(6)+".png")
    bpy.ops.render.render(use_viewport = True, write_still=True)

def get_data(sat, cam, time, distance_scale):
    sat.rotation_mode = 'QUATERNION'
    cam.rotation_mode = 'QUATERNION'
    return np.concatenate([cam.location / distance_scale, wxyz_quat_to_xyzw(sat.delta_rotation_quaternion), wxyz_quat_to_xyzw(cam.rotation_quaternion), [time]])

def wxyz_quat_to_xyzw(wxyz):
    xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    return xyzw
