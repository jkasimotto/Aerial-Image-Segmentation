# Preprocessing RarePlanes Synthetic Data
This script reads in the synthetic masks provided by RarePlanes, converts the mask to a binary mask where 
black = background and white = planes, then tiles the image into smaller segment. Tiles with no planes are
filtered out

## Usage
The script has one mandatory argument with is 'path'. This is the path to a directory
which contains the necessary synthetic data from RarePlanes. The directory should have the following
structure:
```text
path/to/dir
    |--- images
    |--- masks
    |--- xmls
```
### Optional Arg: Tile Size
The size of the tiles can be specified using the `-s` or `--tile-size` argument. The default is 512.

### Optional Arg: Num Workers
The number of processes to use can be specified using the `-n` or `--num-workers` argument. The default is 1.

### Optional Arg: Color Output
Color preserving masks can be generated using the `-c` or `--color` switch.

### Optional Arg: Greyscale Output
Greyscale masks can be generated using the `-g` or `--greyscale` switch.

### Optional Arg: White Output
White preserving masks can be generated using the `-w` or `--white` switch.

## Output Files
A new directory created called `images_tiled` and a new directory created called `greyscale_masks_tiled`,
`color_masks_tiled` or `white_masks_tiled` respective of the switches used.


## Examples
`python preprocessing_tiling.py ./synthetic/train/ -w` \
`python preprocessing_tiling.py ./synthetic/test/ -cgw`
