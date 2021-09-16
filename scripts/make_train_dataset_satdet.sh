#!/bin/bash
data_dir=$1
config=$2
num_vis=200

blend_file=earth/earth.blend

mkdir $data_dir

blender $blend_file --background -noaudio --python simulate_partial.py -- --random_dist --random_translation --bounding_box --label_surfaces --randomise  -v 0.2 -t 0.2 --num_vis $num_vis $config $data_dir
