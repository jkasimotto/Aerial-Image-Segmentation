import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse
import math
from multiprocessing import Process


def extract_plane_colors(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # get the plane colors for an image
    plane_colors = []
    for child in root.findall("object/category0[.='Airplane'].../object_mask_color_rgba"):
        channels = child.text.split(",")
        channels = [int(val) for val in channels]
        plane_colors.append(tuple(channels))

    return plane_colors


def create_mask(image, colors, mode):
    if mode == 'color':
        mask = Image.new('RGB', image.size)
    else:
        mask = Image.new('L', image.size)

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            px_color = image.getpixel((x, y))
            if px_color in colors:
                if mode == 'color':
                    mask.putpixel((x, y), px_color) # preserves the planes' original colors
                elif mode == 'greyscale':
                    mask.putpixel((x, y), colors.index(px_color) + 1) # greyscale planes
                elif mode == 'white':
                    mask.putpixel((x, y), 255) # white planes
            else:
                mask.putpixel((x, y), 0)

    return mask


def tile_img(img, tile_size=512):
    rows, cols = math.ceil(img.height / tile_size), math.ceil(img.width / tile_size)
    total_overlap_x = tile_size * cols - img.width
    total_overlap_y = tile_size * rows - img.height
    min_overlap_x = math.ceil(total_overlap_x / (cols - 1))
    min_overlap_y = math.ceil(total_overlap_y / (rows - 1))
    remainder_x = total_overlap_x % (cols - 1)
    if remainder_x != 0:
        remainder_x = (cols - 1) - remainder_x
    remainder_y = total_overlap_y % (rows - 1)
    if remainder_y != 0:
        remainder_y = (rows - 1) - remainder_y
    overlap_x = ([min_overlap_x] * (cols - 2))
    overlap_x.append(min_overlap_x - remainder_x)
    overlap_x.insert(0, 0)
    overlap_y = ([min_overlap_y] * (rows - 2))
    overlap_y.append(min_overlap_y - remainder_y)
    overlap_y.insert(0, 0)

    tiles = []
    for i in range(rows):
        for j in range(cols):
            start_x, start_y = j * tile_size - sum(overlap_x[:j + 1]), i * tile_size - sum(overlap_y[:i + 1])
            tile = img.crop((start_x, start_y, start_x + tile_size, start_y + tile_size))
            tiles.append(tile)

    return tiles


def process_files(args):
    start = args['ID'] * args['FILES_PER_WORKER']
    end = start + args['FILES_PER_WORKER']
    if args['ID'] == args['NUM_WORKERS'] - 1:
        end = None

    xml_dir = args['XML_DIR']
    tile_size = args['TILE_SIZE']
    output_path = args['OUTPUT_PATH']
    is_color = args['IS_COLOR']
    is_greyscale = args['IS_GREYSCALE']
    is_white = args['IS_WHITE']

    for filename in tqdm(sorted(os.listdir(xml_dir))[start: end], desc=f"Worker {args['ID'] + 1}"):
        xml_file = os.path.join(xml_dir, filename)
        mask_file = os.path.join(xml_dir.replace("xmls", "masks"), filename.replace('.xml', '_mask.png'))
        img_file = os.path.join(xml_dir.replace("xmls", "images"), filename.replace('.xml', '.png'))

        plane_colors = extract_plane_colors(xml_file)
        img = Image.open(img_file)
        img_tiles = tile_img(img, tile_size=tile_size)

        # generate output masks
        mask = Image.open(mask_file) # original mask
        mask_tile_arr = []
        if is_color:
            color_mask = create_mask(mask, plane_colors, 'color')
            color_mask_tiles = tile_img(color_mask, tile_size=tile_size)
            mask_tile_arr.append(color_mask_tiles)
        if is_greyscale:
            greyscale_mask = create_mask(mask, plane_colors, 'greyscale')
            greyscale_mask_tiles = tile_img(greyscale_mask, tile_size=tile_size)
            mask_tile_arr.append(greyscale_mask_tiles)
        if is_white:
            white_mask = create_mask(mask, plane_colors, 'white')
            white_mask_tiles = tile_img(white_mask, tile_size=tile_size)
            mask_tile_arr.append(white_mask_tiles)

        img_dir = 'images_tiled'
        color_mask_dir = 'color_masks_tiled'
        greyscale_mask_dir = 'greyscale_masks_tiled'
        white_mask_dir = 'white_masks_tiled'

        # save tiles to file
        os.makedirs(os.path.join(output_path, img_dir), exist_ok=True)
        if is_color:
            os.makedirs(os.path.join(output_path, color_mask_dir), exist_ok=True)
        if is_greyscale:
            os.makedirs(os.path.join(output_path, greyscale_mask_dir), exist_ok=True)
        if is_white:
            os.makedirs(os.path.join(output_path, white_mask_dir), exist_ok=True)

        for i in range(len(img_tiles)):
            # check if no masks are generated
            if not mask_tile_arr:
                tile_name = filename.replace('.xml', f'_tile_{i}.png')
                img_tiles[i].save(os.path.join(output_path, img_dir, tile_name))
                continue

            # check if mask(s) for a given tile are empty
            img_arr = np.array(mask_tile_arr[0][i])
            n_white_pix = np.sum(img_arr > 0)
            if n_white_pix > 0: # if planes are present
                tile_name = filename.replace('.xml', f'_tile_{i}.png')
                img_tiles[i].save(os.path.join(output_path, img_dir, tile_name))
                if is_color:
                    tile_name = filename.replace('.xml', f'_tile_{i}_color_mask.png')
                    color_mask_tiles[i].save(os.path.join(output_path, color_mask_dir, tile_name))
                if is_greyscale:
                    tile_name = filename.replace('.xml', f'_tile_{i}_greyscale_mask.png')
                    greyscale_mask_tiles[i].save(os.path.join(output_path, greyscale_mask_dir, tile_name))
                if is_white:
                    tile_name = filename.replace('.xml', f'_tile_{i}_white_mask.png')
                    white_mask_tiles[i].save(os.path.join(output_path, white_mask_dir, tile_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        help="path to synthetic train/test data directory. Needs to contain 3 dirs -> images, masks, xmls")
    parser.add_argument("-s", "--tile-size", help="dimension of tile", type=int, default=512)
    parser.add_argument("-n", "--num-workers", help="number of processes utilised", type=int, default=1)
    parser.add_argument("-c", "--color", help="generate color preserving masks", default=False, action='store_true')
    parser.add_argument("-g", "--greyscale", help="generate greyscale masks", default=False, action='store_true')
    parser.add_argument("-w", "--white", help="generate all white masks", default=False, action='store_true')
    args = parser.parse_args()

    xml_dir = os.path.join(args.path, "xmls")
    output_path = os.path.join(args.path)

    num_files = len(os.listdir(xml_dir))
    files_per_worker = num_files // args.num_workers

    PROCESS_ARGS = {
        'NUM_WORKERS': args.num_workers,
        'FILES_PER_WORKER': files_per_worker,
        'XML_DIR': xml_dir,
        'OUTPUT_PATH': output_path,
        'TILE_SIZE': args.tile_size,
        'IS_COLOR': args.color,
        'IS_GREYSCALE': args.greyscale,
        'IS_WHITE': args.white,
    }

    processes = []
    for i in range(args.num_workers):
        PROCESS_ARGS['ID'] = i
        worker_process = Process(target=process_files, args=(PROCESS_ARGS,))
        processes.append(worker_process)
        worker_process.start()

    for process in processes:
        process.join()
