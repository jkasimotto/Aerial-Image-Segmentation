
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as f
from torch import nn
from torchvision.io import ImageReadMode, read_image
from torchvision.models.segmentation import fcn_resnet101
from torchvision.utils import draw_segmentation_masks, make_grid

from model import UNET


def show(images):
    if not isinstance(images, list):
        images = [images]
    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = f.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig("inference.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument(
        "image_dir", help="path to directory containing images to run through the model")
    parser.add_argument("-i", "--index", type=int)
    args = parser.parse_args()

    start, end = 0, args.index
    if args.index is not None:
        start = args.index - 1
        if args.index == 0:
            start, end = 0, 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.model)
    model = UNET(in_channels=3, out_channels=1)
    model.load_state_dict(checkpoint["state_dict"])
    model = nn.DataParallel(model).to(device)
    # model.load_state_dict(checkpoint['model_state_dict']) # This is the new version following FCN.
    # model.load_state_dict(checkpoint['state_dict']) # This is the old version.

    normalisation_factor = 1 / 255

    model.eval()
    masked_images = []
    for filename in os.listdir(args.image_dir)[start: end]:
        # Get image and convert to required format
        img_path = os.path.join(args.image_dir, filename)
        image = read_image(img_path, mode=ImageReadMode.RGB) * \
            normalisation_factor
        image = image.float().unsqueeze(0)

        # Get mask prediction for model
        with torch.inference_mode():
            output = model(image)
            output = torch.sigmoid(output) > 0.5

        image = image * (normalisation_factor ** -1)

        # Draw segmentation mask on top of image
        image = image.squeeze(0).type(torch.uint8)
        image_with_mask = draw_segmentation_masks(
            image=image, masks=output, colors="red")
        masked_images.append(image_with_mask)

    grid = make_grid(masked_images)
    show(grid)


if __name__ == "__main__":
    main()
