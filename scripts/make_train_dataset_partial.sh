#!/bin/bash
data_dir=$1
config=$2
num_vis=50

blend_file=earth/earth.blend

mkdir $data_dir

blender $blend_file --background -noaudio --python simulate_partial.py -- --label_surfaces --randomise --pair_images -v 0.0 -t 0.0 --num_vis $num_vis $config $data_dir
