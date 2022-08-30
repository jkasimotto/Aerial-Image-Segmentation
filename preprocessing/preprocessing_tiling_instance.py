import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse
import math
from multiprocessing import Process


def extract_plane_colors(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # get the plane colors for an image
    plane_colors = []
    for child in root.findall("object/category0[.='Airplane'].../object_mask_color_rgba"):
        channels = child.text.split(",")
        channels = [int(val) for val in channels]
        plane_colors.append(tuple(channels))

    return plane_colors


def create_mask(image, colors, is_colored, greyscale_boost):
    if is_colored:
        new_mask = Image.new('RGB', image.size)
    else:
        new_mask = Image.new('L', image.size)

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            px_color = image.getpixel((x, y))
            if px_color in colors:
                if is_colored:
                    new_mask.putpixel((x, y), px_color) # preserves the planes' original colors
                else:
                    new_mask.putpixel((x, y), colors.index(px_color) + greyscale_boost) # greyscale planes
            else:
                new_mask.putpixel((x, y), 0)

    return new_mask


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
    is_colored = args['IS_COLORED']
    greyscale_boost = args['GREYSCALE_BOOST']

    for filename in tqdm(sorted(os.listdir(xml_dir))[start: end], desc=f"Worker {args['ID'] + 1}"):
        xml_file = os.path.join(xml_dir, filename)
        mask_file = os.path.join(xml_dir.replace("xmls", "masks"), filename.replace('.xml', '_mask.png'))
        img_file = os.path.join(xml_dir.replace("xmls", "images"), filename.replace('.xml', '.png'))

        plane_colors = extract_plane_colors(xml_file)
        img = Image.open(img_file)
        mask = create_mask(Image.open(mask_file), plane_colors, is_colored, greyscale_boost)

        img_tiles = tile_img(img, tile_size=tile_size)
        mask_tiles = tile_img(mask, tile_size=tile_size)

        # save tiles to file
        os.makedirs(os.path.join(output_path, 'instance_images_tiled'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'instance_masks_tiled'), exist_ok=True)
        for i in range(len(img_tiles)):
            img_arr = np.array(mask_tiles[i])
            n_white_pix = np.sum(img_arr > 0)
            if n_white_pix > 0:
                tile_name = filename.replace('.xml', f'_tile_{i}.png')
                img_tiles[i].save(os.path.join(output_path, 'instance_images_tiled', tile_name))
                tile_name = filename.replace('.xml', f'_tile_{i}_mask.png')
                mask_tiles[i].save(os.path.join(output_path, 'instance_masks_tiled', tile_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        help="path to synthetic train/test data directory. Needs to contain 3 dirs -> images, masks, xmls")
    parser.add_argument("-s", "--tile-size", help="dimension of tile", type=int, default=512)
    parser.add_argument("-n", "--num-workers", help="number of processes utilised", type=int, default=1)
    parser.add_argument("-c", "--color", help="if set, masks are coloured", default=False, action='store_true')
    parser.add_argument("-b", "--greyscale-boost", help="base alpha of greyscale masks", type=int, default=0)
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
        'IS_COLORED': args.color,
        'GREYSCALE_BOOST': args.greyscale_boost,
    }

    processes = []
    for i in range(args.num_workers):
        PROCESS_ARGS['ID'] = i
        worker_process = Process(target=process_files, args=(PROCESS_ARGS,))
        processes.append(worker_process)
        worker_process.start()

    for process in processes:
        process.join()
