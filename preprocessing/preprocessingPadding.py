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


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_files(args):
    start = args['ID'] * args['FILES_PER_WORKER']
    end = start + args['FILES_PER_WORKER']
    if args['ID'] == args['NUM_WORKERS'] - 1:
        end = None

    xml_dir = args['XML_DIR']
    pad_size = args["PAD_SIZE"]
    output_path = args["OUTPUT_PATH"]

    for filename in tqdm(sorted(os.listdir(xml_dir))[start: end], desc=f"Worker {args['ID'] + 1}"):
        xml_file = os.path.join(xml_dir, filename)
        mask_file = os.path.join(xml_dir.replace("xmls", "masks"), filename.replace('.xml', '_mask.png'))
        img_file = os.path.join(xml_dir.replace("xmls", "images"), filename.replace('.xml', '.png'))

        plane_colors = extract_plane_colors(xml_file)

        # full_mask = full sized rareplanes mask
        full_mask = Image.open(mask_file)
        new_mask = create_mask(full_mask, plane_colors)
        og_img = Image.open(img_file)

        # remove blank tiles
        # img_tiles, mask_tiles = filter_tiles(tile_img(og_img, tile_size=tile_size),
        #                                      tile_img(new_mask, tile_size=tile_size))
        # img_pad, mask_pad = pad_images(og_img, new_mask, pad_size=pad_size)
        img_pad = expand2square(og_img, 0)
        mask_pad = expand2square(new_mask, 0)

        # assert (len(img_pad) == len(mask_pad))

        # save tiles to file
        os.makedirs(os.path.join(output_path, 'images_padded'), exist_ok=True)
        pad_name = filename.replace('.xml', '_padded.png')
        img_pad.save(os.path.join(output_path, 'images_padded', pad_name))


        os.makedirs(os.path.join(output_path, 'masks_padded'), exist_ok=True)
        pad_name = filename.replace('.xml', '_padded_mask.png')
        mask_pad.save(os.path.join(output_path, 'masks_padded', pad_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        help="path to synthetic train/test data directory. Needs to contain 3 dirs -> images, masks, xmls")
    parser.add_argument("-s", "--pad-size", help="dimension of tile", type=int, default=2048)
    parser.add_argument("-n", "--num-workers", help="number of processes utilised", type=int, default=1)
    args = parser.parse_args()

    xml_dir = os.path.join(args.path, "xmls")
    output_path = args.path

    num_files = len(os.listdir(xml_dir))
    files_per_worker = num_files // args.num_workers

    PROCESS_ARGS = {
        'NUM_WORKERS': args.num_workers,
        'FILES_PER_WORKER': files_per_worker,
        'XML_DIR': xml_dir,
        'OUTPUT_PATH': output_path,
        'PAD_SIZE': args.pad_size,
    }

    processes = []
    for i in range(args.num_workers):
        PROCESS_ARGS['ID'] = i
        worker_process = Process(target=process_files, args=(PROCESS_ARGS,))
        processes.append(worker_process)
        worker_process.start()

    for process in processes:
        process.join()
