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
import bpy
from satvis.utils.blender_utils import make_sat, rotate_sat, save_render, get_sat_path
from satvis.utils.data_utils import make_directory

def label_surfaces(configs, labels_dir, sat_initial_attitude, rotation_step):
    """
    Replace the target with the labelled model for each label directory. Only the new model requires
    animation frames since the camera position and direction has been defined already.
    """
    # For each label .obj file, add labels to a directory with the name of the label
    label_names = []
    for filename in os.listdir(configs["satObjDir"]):
        if configs["target"] + "-label" in filename and not filename.endswith(".mtl"):
            label = "-".join(os.path.splitext(filename)[0].split("-")[2:])
            label_names.append(label)

    for label in label_names:
        label_filepath = os.path.join(labels_dir, label)
        make_directory(label_filepath)
        sat_path = get_sat_path(configs["satObjDir"], configs["target"] + "-label-" + label)

        # Make object
        sat = make_sat(sat_path, sat_initial_attitude, configs["distance_scale"])
        for mat in sat.data.materials:
            mat.use_shadeless = True

        # Animation
        frame_end = configs["duration"] * configs["fps"]
        bpy.context.scene.frame_end = configs["duration"] * configs["fps"]

        # Rotate satellite by constant angular velocity
        for i in range(frame_end//configs["frame_skip"]+1):
            rotate_sat(sat, i*configs["frame_skip"], rotation_step)

        # Save animation
        for i in range(frame_end//configs["frame_skip"]+1):
            bpy.context.scene.frame_set(i*configs["frame_skip"])
            save_render(i, label_filepath)

        # Remove satellite object
        sat.select = True
        for child in sat.children:
            child.select = True
        bpy.ops.object.delete()
