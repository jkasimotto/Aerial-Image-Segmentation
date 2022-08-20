import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse


def extract_plane_colors(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # get the plane colors for an image
    plane_colors = []
    for child in root.findall("object/category0[.='Airplane'].../object_mask_color_rgba"):
        channels = child.text.split(",")
        channels = [int(val) for val in channels]
        plane_colors.append(channels)

    return plane_colors


def is_plane_color(px, colors):
    for color in colors:
        if px[0] == color[0] and px[1] == color[1] and px[2] == color[2] and px[3] == color[3]:
            return True
    return False


def create_mask(image, colors):
    new_mask = Image.new('L', image.size)

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            px_color = image.getpixel((x, y))
            if is_plane_color(px_color, colors):
                new_mask.putpixel((x, y), 255)
            else:
                new_mask.putpixel((x, y), 0)

    return new_mask


def tile_img(img, tile_size=512):
    rows, cols = img.height // tile_size, img.width // tile_size
    resized_width = cols * tile_size
    resized_height = rows * tile_size
    resized = img.resize((resized_width, resized_height))
    tiles = []
    for i in range(rows):
        for j in range(cols):
            start_x, start_y = j * tile_size, i * tile_size
            tile = resized.crop((start_x, start_y, start_x + tile_size, start_y + tile_size))
            tiles.append(tile)
    return tiles


def filter_tiles(img_tiles, mask_tiles):
    filtered_img_tiles = []
    filtered_mask_tiles = []
    for i in range(len(mask_tiles)):
        img_arr = np.array(mask_tiles[i])
        n_white_pix = np.sum(img_arr > 0)
        if n_white_pix > 0:
            filtered_mask_tiles.append(mask_tiles[i])
            filtered_img_tiles.append(img_tiles[i])

    return filtered_img_tiles, filtered_mask_tiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        help="path to synthetic train/test data directory. Needs to contain 3 dirs -> images, masks, xmls")
    parser.add_argument("-s", "--tile-size", help="dimension of tile", type=int, default=512)
    args = parser.parse_args()

    xml_dir = os.path.join(args.path, "xmls")
    output_path = args.path

    for filename in tqdm(sorted(os.listdir(xml_dir))):
        xml_file = os.path.join(xml_dir, filename)
        mask_file = os.path.join(xml_dir.replace("xmls", "masks"), filename.replace('.xml', '_mask.png'))
        img_file = os.path.join(xml_dir.replace("xmls", "images"), filename.replace('.xml', '.png'))

        plane_colors = extract_plane_colors(xml_file)

        # full_mask = full sized rareplanes mask
        full_mask = Image.open(mask_file)
        new_mask = create_mask(full_mask, plane_colors)
        og_img = Image.open(img_file)

        # remove blank tiles
        img_tiles, mask_tiles = filter_tiles(tile_img(og_img, tile_size=args.tile_size),
                                             tile_img(new_mask, tile_size=args.tile_size))

        assert (len(img_tiles) == len(mask_tiles))

        # save tiles to file
        os.makedirs(os.path.join(output_path, 'images_tiled'), exist_ok=True)
        for tile in img_tiles:
            tile_name = filename.replace('.xml', f'_tile_{img_tiles.index(tile) + 1}.png')
            tile.save(os.path.join(output_path, 'images_tiled', tile_name))

        os.makedirs(os.path.join(output_path, 'masks_tiled'), exist_ok=True)
        for tile in mask_tiles:
            tile_name = filename.replace('.xml', f'_tile_{mask_tiles.index(tile) + 1}_mask.png')
            tile.save(os.path.join(output_path, 'masks_tiled', tile_name))
