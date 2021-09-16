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
import yaml
import shutil
import numpy as np
import mathutils
from satvis.utils.blender_argparse import ArgumentParserForBlender
from satvis.utils.label_surfaces import label_surfaces
from satvis.utils.bounding_box import write_bounds_2d
from satvis.utils.vis_utils import setup_scene, setup_params_partial, animate
from satvis.utils.data_utils import rename_depth_files, clip_depth, save_vis_to_dir, make_data_subsets, setup_directories, get_change_in_attitude
from satvis.utils.blender_utils import CAMERA_VIEW_DIRECTION

def main(initial_configs, dirname, params):
    # Get all debris models from directory
    all_targets = [os.path.splitext(f)[0] for f in os.listdir(initial_configs["satObjDir"])
                    if os.path.isfile(os.path.join(initial_configs["satObjDir"], f)) and "-label" not in f] # Get all debris models from directory
    assert params.target in all_targets + [None], "Target not in available targets"

    # Run all simulations and collect data
    for vis_iter in range(params.num_vis):
        # Randomise parameters
        if params.randomise:
            configs = randomise_configs(initial_configs.copy(), params.target)
        else:
            configs = initial_configs.copy()
        if not isinstance(configs['cam_pos'], list): # If only a distance is provided, choose a random direction
            direction = np.random.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            configs['cam_pos'] = direction * configs['cam_pos']
        if params.random_dist:
            configs['cam_pos'] = np.array(configs['cam_pos']) * (np.random.rand() + 0.5) # Between 0.5 and 1.5 times the provided value
        if configs['cam_dir'] is None: # If direction is not provided, face towards the target
            configs['cam_dir'] = -np.array(configs['cam_pos']) / np.linalg.norm(configs['cam_pos'])
        if params.random_translation:
            dist = np.linalg.norm(configs['cam_pos'])
            max_translation = np.maximum(dist * np.tan(np.pi/180*configs['fov']/2.) - 10, 0) # Assuming target is scaled to 10m
            translation_view = mathutils.Vector(list((2*np.random.rand(2)-1) * max_translation) + [0.])
            # Convert translation from camera coords to world coords
            view_angle = np.arccos(np.dot(configs['cam_dir'], CAMERA_VIEW_DIRECTION))
            view_axis = np.cross(CAMERA_VIEW_DIRECTION, configs['cam_dir'])
            view_quat = mathutils.Quaternion(view_axis, view_angle)
            translation_world = translation_view.copy()
            translation_world.rotate(view_quat.conjugated())
            configs['cam_pos'] = configs['cam_pos'] + translation_world
        configs['target'] = params.target
        if params.target is None:
            if isinstance(configs['satellites'], list):
                configs['target'] = np.random.choice(configs['satellites'])
            else:
                configs['target'] = np.random.choice(all_targets)

        # Make directories
        dirs = setup_directories(dirname)

        # Visualise scene
        setup_scene(configs)
        rotation_step, earth, sat, sun, cam, torch = setup_params_partial(configs, params, dirs["tmp_depth"])
        animate(configs, params, dirs["tmp_img"], dirs["root"], rotation_step, earth, sat, sun, cam, torch, configs["period"])

        # Label bounding box
        if params.bounding_box:
            boundingbox_filepath = os.path.join(dirs["data"], "bounding_box.txt")
            frame_end = configs["duration"] * configs["fps"]
            write_bounds_2d(boundingbox_filepath, bpy.context.scene, cam, sat, configs["frame_skip"], frame_end, configs["target"])

        # Label surfaces
        if params.label_surfaces:
            # Remove objects from scene
            for obj in [earth, sat, sun, torch]:
                obj.select = True
                for child in obj.children:
                    child.select = True
            bpy.ops.object.delete()
            sat_initial_attitude = mathutils.Euler(configs["attitude"], 'XYZ').to_quaternion()
            label_surfaces(configs, dirs["tmp_labels"], sat_initial_attitude, rotation_step)

        # Rename depth files due to blender using the frame number to name files
        rename_depth_files(dirs["tmp_depth"], configs)
        # Clip the saved depth files so that the background has depth of 0
        clip_depth(dirs["tmp_depth"], configs)
        # Save the result of this visualisation to the directory and clear temporary vis dirs
        save_vis_to_dir(dirs)
        shutil.rmtree(dirs["tmp_img"])
        shutil.rmtree(dirs["tmp_depth"])
        shutil.rmtree(dirs["tmp_labels"])
        os.remove(os.path.join(dirs["root"], "data.txt"))

        # Remove all loaded objects and data from blender
        bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)

    # Divide the data into subsets
    make_data_subsets(dirs["img"], dirs["subsets"], False, params.val_fraction, params.test_fraction, params.shuffle)
    # Collect into image pairs
    if params.pair_images:
        make_data_subsets(dirs["img"], dirs["subsets"], True, params.val_fraction, params.test_fraction, params.shuffle)
        get_change_in_attitude(dirs["img"], dirs["data"], dirs["subsets"])

def randomise_configs(configs, target=None):
    configs['attitude'] = (np.random.rand(3) - 0.5) * 2 * 20.
    configs['ang_vel'] = 2 + (np.random.rand() * 8) # Rotation between 2 and 10 degrees
    axis = np.random.normal(size=3)
    configs['axis'] = axis / np.linalg.norm(axis)
    configs['sun_dir'] = np.random.normal(size=3)
    configs['sun_dir'] = configs['sun_dir'] / np.linalg.norm(configs['sun_dir'])
    return configs

def parse_args():
    def restricted_float(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise ValueError("%r not in range [0.0, 1.0]"%(x,))
        return x

    parser = ArgumentParserForBlender(description='Generate synthetic data using blender')
    parser.add_argument('configs', help='Config file (YAML)')
    parser.add_argument('dir', help='Path to the directory to save image files')
    parser.add_argument('-r', '--randomise', default=False, action='store_true',
                        help='Randomise configs')
    parser.add_argument('--random_dist', default=False, action='store_true',
                        help='Randomise the initial separation between target and chaser')
    parser.add_argument('--random_translation', default=False, action='store_true',
                        help='Add a random translation such that the target is not in the centre of the image')
    parser.add_argument('--sun_behind', default=False, action='store_true',
                        help='Restrict the sun to be behind the camera')
    parser.add_argument('-l', '--label_surfaces', default=False, action='store_true',
                        help='Label the surfaces of the satellite')
    parser.add_argument('-b', '--bounding_box', default=False, action='store_true',
                        help='Label the bounding box around the satellite')
    parser.add_argument('-p', '--pair_images', default=False, action='store_true',
                        help='Collect into image pairs and label with the change in attitudes over the timestep')
    parser.add_argument('--no_torch', default=False, action='store_true',
                        help='No torch on chaser')
    parser.add_argument('--target', default=None,
                        help='Name of the target satellite model. If not provided, use a random one from those in the model directory')
    parser.add_argument('-n', '--num_vis', default=1, type=int,
                        help='Number of visualisations to simulate')
    parser.add_argument("-v", "--val_fraction", type=restricted_float, default=0.1,
                        help="Fraction of images to use for validation")
    parser.add_argument("-t", "--test_fraction", type=restricted_float, default=0.1,
                        help="Fraction of images to use for testing")
    parser.add_argument("-s", "--shuffle", default=False, action="store_true",
                        help="Whether to shuffle the subsets")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.configs) as file:
        yaml_configs = yaml.load(file)

    main(yaml_configs, args.dir, args)
