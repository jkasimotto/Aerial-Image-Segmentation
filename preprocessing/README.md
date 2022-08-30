# Preprocessing Rareplanes Synthetic Data
This script reads in the synthetic masks provided by Rareplanes, converts the mask to a binary mask where 
black = background and white = planes, then tiles the image into smaller segment. Tiles with no planes are
filtered out

## Usage
The script has one mandatory argument with is 'path'. This is the path to a directory
which contains the necessary synthetic data from Rareplanes. The directory should have the following
structure:
```text
path/to/dir
    |--- images
    |--- masks
    |--- xmls
```
### Optional Arg: Tile Size
The size of the tiles can be specified using the `-s` or `--tile-size` arguments. The default is 512.

### Optional Arg: Num Workers
The number of processes to use can be specified using the `-n` or `--num-workers` arguments. The default is 1.

### Optional Arg For Instance Version: Color
Color in the outputed masks can be enabled using the `-c` or `--color` switches. Ommiting the switch defaults
to greyscale output.

### Optional Arg For Instance Version: Greyscale Boost
The base brightness of the planes in the greyscale masks can be specified using the `-b` or `--greyscale-boost`
arguments. The default is 0.

## Output Files
Two new directories are created called `images_tiled` and `masks_tiled` or `instance_images_tiled` and
`instance_masks_tiled` for the instance version.
