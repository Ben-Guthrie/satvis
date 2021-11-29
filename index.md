# satvis: On-orbit visualisation of satellite proximity operations
The aim of this code is to construct realistic datasets of image data in on-orbit proximity operations. This work attempts to facilitate research into applications of machine learning to the problem of visual guidance in proximity operations in space.

## Setup
- Install blender 2.79
- Setup blender's python environment as explained below
- Add satvis directory to PYTHONPATH

### Install python dependencies
```
cd </path/to/blender>/python/bin
./python -m ensurepip
./python -m pip install --upgrade pip
./python -m pip install scikit-build
./python -m pip install opencv-python numpy pyyaml
```

## Usage
See examples in scripts/ to generate datasets.
To run a simulation, start blender from the command line and pass the blend file (earth/earth.blend) and the python file, i.e.
```
blender earth/earth.blend --background -noaudio --python simulate_full.py -- --help
```
Blender command line arguments are separated from the python command line arguments by the two dashes alone, "--".
The generated data can be controlled using the command line arguments or by changing the parameters in the config files.
See configs/ for example config files.
For more information on the arguments, use --help.

If a dataset already exists in the directory, this will be appended to instead of replaced.

### Simulation modes
There are two possible modes in which to run the simulation. The full simulation (accessed by passing the file simulate_full.py to blender) accurately simulates the relative motion between two nearby satellites in proximity operations in orbit. On the other hand, the partial simulation (simulate_partial.py) is useful for generating data under certain specific conditions, by providing more control over the parameters of the visualisation, but does not necessarily undergo realistic on-orbit dynamics.

## Generated datasets
The datasets have the following format, assuming all data labels are used:

dataset_name
- data
  - all_data.txt
  - bounding_box.txt
  - rotations.txt
- depth
  - 000000.exr, 000001.exr, ...
- img
  - 000000.png, 000001.png, ...
- labels
  - body
    - 000000.png, 000001.png, ...
  - solar panel
    - 000000.png, 000001.png, ...
  - ...
- subsets
  - image_subsets.txt
  - pair_subsets.txt

Note that in the datasets, the image index is skipped after each simulation ends. For example, if a simulation generates 50 image files, labelled 000000.png to 000049.png, then index 50 will be skipped before inserting the data from the next simulation, which will instead start with 000051.png.

The subsets/ directory contains a subset label, either 0, 1 or 2 for "train", "valid" and "test". In image_subsets.txt, this is given for each image file, whereas pair_subsets.txt collects the files into image pairs with different separations between images, and assigns each to a subset. This latter option is relevant when looking at change in positions or attitudes over time.

### Data labels
In data/all_data.txt, the images are each labelled, in the following order, with the relative position (x, y, z) of the chaser, quaternion describing the attitude of the target (x, y, z, w), quaternion of the chaser (x, y, z, w) and time (t). The line number corresponds to the image number, where the first line contains the data for image 000000.png.

If --bounding_box is selected, the file data/bounding_box.txt contains the name and bounding box about the satellite. The bounding box is defined as (x, y, w, h), where x and y are the pixel coordinates of the top-left of the box, and w and h are the width and height in pixels.

If --label_surfaces is selected, then labels/ contains a separate directory for all surface labels provided in the debrismodels directory. If a satellite model has a corresponding surface label model (named <model_name>-label-<label_name>) then the label is stored in labels/<label_name>. This label is a black and white image, where a pixel with intensity > 0.5 corresponds to the specific surface.

If --pair_images is selected, the data is collected into pairs of images, with different separations between the image files. For each pair of images, the file data/rotations.txt contains the index of the "before" and "after" images, the observed rotation quaternion (x, y, z, w) which is a combination of rotations of both target and chaser, and the scalar distance to the target before and after the timestep.

## Adding new satellite models
The simulation can access any 3D models placed in the debrismodels/ directory. Currently, the software can load .obj, .fbx, .x3d and .3ds files. More satellites can be added to the datasets by simply adding the model file to the directory. If the model is desired to be labelled with surface segmentations, then it is necessary to also provide a black and white model, where the white surfaces correspond to the label, in the format <model_name>-label-<label_name>.

## Pre-made datasets
A number of training and test datasets are available at https://www.kaggle.com/benguthrie/inorbit-satellite-image-datasets
