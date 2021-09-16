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
import os.path as osp
import random
import cv2
import numpy as np
import shutil
from shutil import copyfile
from satvis.orbital_dynamics.attitude_conversion import get_rotation_arb_frame

PATHS = {
"root": "",
"tmp_img": "frames",
"tmp_depth": "depth_frames",
"tmp_labels": "label_frames",
"img": "img",
"depth": "depth",
"labels": "labels",
"data": "data",
"subsets": "subsets"
}

def setup_directories(dirname):
    # Make directories
    dirname = osp.abspath(dirname)
    all_dirs = {}
    for key, value in PATHS.items():
        path = osp.join(dirname, value)
        make_directory(path)
        all_dirs[key] = path
    return all_dirs


def make_directory(dirname, remove_old=False):
    if osp.isdir(dirname):
        if remove_old:
            shutil.rmtree(dirname)
        else:
            return
    os.mkdir(dirname)

def rename_depth_files(depth_dir, configs):
    all_files = os.listdir(depth_dir)
    all_files.sort()
    for depthFile in all_files:
        if osp.isfile(osp.join(depth_dir, depthFile)):
            filenum = int(osp.splitext(osp.basename(depthFile))[0])
            filenum = filenum // configs["frame_skip"]
            new_filename = osp.join(depth_dir, str(filenum).zfill(6)+'.exr')
            if filenum > 0:
                assert(not osp.isfile(new_filename))
            os.rename(osp.join(depth_dir, depthFile), new_filename)

def clip_depth(depth_dir, configs):
    all_files = os.listdir(depth_dir)
    all_files.sort()
    for depthFile in all_files:
        if osp.isfile(osp.join(depth_dir, depthFile)):
            depth = cv2.imread(osp.join(depth_dir, depthFile), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth[depth > configs["max_depth"] * configs["distance_scale"]] = 0.
            depth = depth / configs["distance_scale"]
            cv2.imwrite(osp.join(depth_dir, depthFile), depth)

def save_vis_to_dir(dirs):
    frame_dir = dirs["tmp_img"]
    img_dir = dirs["img"]

    dir_list = [f for f in os.listdir(img_dir) if osp.isfile(osp.join(img_dir, f)) and f.endswith('.png')]

    # If the directory already contains image files, start numbering from the final number instead of 0
    if dir_list:
        dir_list.sort()
        final_num = int(dir_list[-1].split('.')[0])
        offset = final_num + 2  # Skip one index so that frames from different videos are not paired together
    else:
        offset = 0

    # Copy files from frames directory to img directory
    images = [f for f in os.listdir(frame_dir) if osp.isfile(osp.join(frame_dir, f)) and f.endswith('.png')]
    images.sort()

    for i, img in enumerate(images):
        idx = i + offset
        fname = osp.join(img_dir, str(idx).zfill(6)+'.png')
        copyfile(osp.join(frame_dir, img), fname)

    # Copy files from depth_frames directory to depth directory
    if osp.exists(dirs["tmp_depth"]):
        copy_from_frame_dir(dirs["tmp_depth"], dirs["depth"], offset)

    # Copy files from label_frames directory to labels directory
    if osp.exists(dirs["tmp_labels"]):
        copy_from_frame_dir(dirs["tmp_labels"], dirs["labels"], offset, subdirs=True)

    append_positional_data(dirs["root"], dirs["data"])


def copy_from_frame_dir(frame_dir, target_dir, offset, subdirs=False):
    if not osp.exists(target_dir):
        os.makedirs(target_dir)
    if subdirs:
        dirs = [d for d in os.listdir(frame_dir) if osp.isdir(osp.join(frame_dir, d))]
        for d in dirs:
            copy_from_frame_dir(osp.join(frame_dir, d), osp.join(target_dir, d), offset, subdirs=False)
        return
    images = [f for f in os.listdir(frame_dir) if osp.isfile(osp.join(frame_dir, f))]
    if images:
        ext = os.path.splitext(images[0])[1]
        images.sort()

        for i, img in enumerate(images):
            idx = i + offset
            fname = os.path.join(target_dir, str(idx).zfill(6)+ext)
            copyfile(os.path.join(frame_dir, img), fname)


def append_positional_data(root_dir, data_dir):
    data_file = os.path.join(root_dir, 'data.txt')
    if os.path.isfile(data_file):
        add_line = os.path.exists(os.path.join(data_dir, 'all_data.txt'))
        with open(os.path.join(data_dir, 'all_data.txt'), 'a') as fw:
            if add_line:
                fw.write('\n')
            with open(data_file, 'r') as fr:
                fw.writelines(fr.readlines()[1:])
    else:
        print("Cannot find positional data. Ignoring")

def make_data_subsets(img_path, subsets_path, pairs=False, val_fraction=0.1, test_fraction=0.1, shuffle=True):
    """
    Create a txt file in the 'subsets' folder containing either the image or image pair and the subset.
    """
    images_list = [f for f in os.listdir(img_path) if osp.isfile(osp.join(img_path, f)) and f.endswith('.png')]
    images_list.sort()

    if pairs:
        # Repeat for n separations between frames, from 1 to 4
        data_subsets = []
        for i in range(1, 5):
            future_image_files = images_list[i:]
            image_files = images_list[:-i]
            img_asint = [int(f[:6]) for f in image_files]

            count_offset = 0
            data_offset = 0
            for j, img_num in enumerate(img_asint):
                if j + count_offset != img_num:
                    count_offset += 1
                    for k in range(i):
                        data_offset += 1
                        image_files.pop(j-data_offset)
                        future_image_files.pop(j-data_offset)
                        #print("removed img:", image_files.pop(j-data_offset))
                        #print("removed future img:", future_image_files.pop(j-data_offset))

            nframes = len(image_files)
            data_set = np.zeros_like(image_files, dtype=int)
            train_cutoff = int(nframes*(1.0 - val_fraction - test_fraction))
            val_cutoff = int(nframes*(1.0 - test_fraction))
            data_set[train_cutoff:val_cutoff] = 1
            data_set[val_cutoff:] = 2
            if shuffle:
                random.shuffle(data_set)

            data_structure = np.empty(len(image_files), dtype=[('img', 'U10'), ('future', 'U10'), ('subset', int)])
            data_structure['img'] = image_files
            data_structure['future'] = future_image_files
            data_structure['subset'] = data_set

            data_subsets.append(data_structure)

        data_subsets = np.concatenate(data_subsets)
        data_subsets.sort(order=('img', 'future'))
        header='image \t future image \t subset'

        if not os.path.exists(subsets_path):
            os.makedirs(subsets_path)

        np.savetxt(osp.join(subsets_path, 'pair_subsets.txt'), data_subsets, fmt='%s', header=header)

    # Make subsets for single images
    else:
        data_set = np.zeros_like(images_list, dtype=np.int)
        nframes = len(data_set)
        train_cutoff = int(nframes*(1.0 - val_fraction - test_fraction))
        val_cutoff = int(nframes*(1.0 - test_fraction))
        data_set[train_cutoff:val_cutoff] = 1
        data_set[val_cutoff:] = 2
        data_structure = np.array([(images_list[i], data_set[i]) for i in range(nframes)], dtype=[('image', 'U10'), ('subset', int)])
        header='image \t subset'
        np.savetxt(osp.join(subsets_path, 'image_subsets.txt'), data_structure, fmt='%s', header=header)


def get_change_in_attitude(img_path, data_path, subsets_path):
    fname = os.path.join(data_path, "all_data.txt")
    all_data = np.loadtxt(fname)
    is_3d_data = all_data.shape[1] > 10
    if is_3d_data:
        target_q = all_data[:, 3:7]
        chaser_q = all_data[:, 7:11]
    else:
        angles = all_data[:, 3]
    pos = all_data[:,:3]

    # Get the missing frames in the saved images for indexing
    images_list = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.endswith('.png')]
    images_list.sort()
    img_indices = [int(img[:6]) for img in images_list]
    missing_frames = []
    offset = 0
    for i in range(img_indices[-1]):
        if i != img_indices[i - offset]:
            missing_frames.append(i)
            offset += 1

    subsets_fname = os.path.join(subsets_path, "pair_subsets.txt")
    img_subsets = np.loadtxt(subsets_fname, skiprows=1, dtype='str')

    if is_3d_data:
        rotated_abs_dq = np.zeros((img_subsets.shape[0], 4))
        #abs_dq = np.zeros_like(rotated_abs_dq)
    else:
        rotations = np.zeros(img_subsets.shape[0])
    positions = np.zeros((img_subsets.shape[0], 2))

    images = [int(img[:6]) for img in img_subsets[:, 0]]
    future_images = [int(img[:6]) for img in img_subsets[:, 1]]

    missed_frames = 0
    # Get the quaternion between each of the pairs of image frames
    for i in range(img_subsets.shape[0]):
        if missing_frames and images[i] == missing_frames[0] + 1:
            missed_frames += 1
            del missing_frames[0]
        current_idx = images[i] - missed_frames
        future_idx = future_images[i] - missed_frames

        positions[i] = [np.linalg.norm(pos[current_idx]), np.linalg.norm(pos[future_idx])]
        if is_3d_data:
            rotated_abs_dq[i] = get_rotation_arb_frame(target_q[current_idx],
                                                       target_q[future_idx],
                                                       chaser_q[current_idx],
                                                       chaser_q[future_idx])
        else:
            rotations[i] = angles[future_idx] - angles[current_idx]
    if is_3d_data:
        outputs = rotated_abs_dq
    else:
        outputs = rotations


    output_qs = np.c_[images, future_images, np.array(outputs), np.array(positions)]
    fname = os.path.join(data_path, "rotations.txt")
    np.savetxt(fname, np.asarray(output_qs), header="image before, image after, rotation (x, y, z, w), distance before, distance after")
