#!/bin/bash
#angvels=(0.5 1 1.5 2 3 4 5 6 7)
angvels=(5)
#sats=('ICECube' 'LRO' 'MiRaTA' 'ICESat2' 'Calipso' 'CloudSat')
#sats=('Aqua' 'Sentinel6' 'SolarB')
sats=('Aqua' 'Sentinel6' 'SolarB' 'ICECube' 'LRO' 'MiRaTA' 'ICESat2' 'Calipso' 'CloudSat')

axis="[3, 2, -4]"

data_dir=$1
config=$2

tmp_config=tmp_config_satdet.yaml
num_vis=1
blend_file=earth/earth.blend

mkdir $data_dir
cp $config $tmp_config

# Edit temporary config file for each simulation case
for avel in "${angvels[@]}"; do
	# Change specific lines of the temporary config file
	sed -i "17s/.*/ang_vel : ${avel}/" $tmp_config
	sed -i "18s/.*/axis : ${axis}/" $tmp_config
	for sat in "${sats[@]}"; do
		avel_label=$(sed 's/\./-/g' <<<"$avel") # Replace e.g. 0.5 with 0-5 in filename

		dirname=$data_dir/test_data_${avel_label}_${sat}
		rm -rf $dirname
		mkdir $dirname

		# Simulate
		blender $blend_file --background --python simulate_partial.py -- --label_surfaces --target $sat -v 0.0 -t 1.0 --num_vis $num_vis $tmp_config $dirname --bounding_box
	done
done

rm $tmp_config
