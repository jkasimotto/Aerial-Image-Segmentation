import torch
from torch import nn
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_segmentation_masks, make_grid
import argparse
import os
import numpy as np
import torchvision.transforms.functional as f
import matplotlib.pyplot as plt


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = f.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument("image_dir", help="path to directory containing images to run through the model")
    args = parser.parse_args()

    model = torch.load(args.model)

    # Needed for eval to work properly for some reason
    # Batch norm layers were doing something funny
    for m in model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None

    masked_images = []
    for filename in os.listdir(args.image_dir):
        # Get image and convert to required format
        img_path = os.path.join(args.image_dir, filename)
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = image.float().unsqueeze(0)

        # Get mask prediction for model
        model.eval()
        with torch.inference_mode():
            output = model(image)['out']
            output = output.softmax(dim=1).argmax(dim=1) > 0

        # Draw segmentation mask on top of image
        image = image.squeeze(0).type(torch.uint8)
        image_with_mask = draw_segmentation_masks(image=image, masks=output, colors="red")
        masked_images.append(image_with_mask)

    grid = make_grid(masked_images)
    show(grid)


if __name__ == "__main__":
    main()
