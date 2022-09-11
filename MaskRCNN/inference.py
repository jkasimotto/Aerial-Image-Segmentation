import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_segmentation_masks, make_grid
import argparse
import os
import numpy as np
import torchvision.transforms.functional as f
import matplotlib.pyplot as plt
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as MaskRCNN


def show(images):
    if not isinstance(images, list):
        images = [images]
    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = f.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument("image_dir", help="path to directory containing images to run through the model")
    parser.add_argument("-i", "--index", type=int)
    args = parser.parse_args()

    start, end = 0, args.index
    if args.index is not None:
        start = args.index - 1
        if args.index == 0:
            start, end = 0, 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.model)
    num_classes = 2
    model = MaskRCNN(
            weights=None,
            num_classes=num_classes, # optional
            weights_backbone=None)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    normalisation_factor = 1 / 255

    # evaluation mode
    model.eval()
    masked_images = []
    for filename in os.listdir(args.image_dir)[start: end]:
        # Get image and convert to required format
        img_path = os.path.join(args.image_dir, filename)
        image = read_image(img_path, mode=ImageReadMode.RGB) * normalisation_factor
        image = image.float().unsqueeze(0)
        
        # Get mask prediction for model
        with torch.inference_mode():
            output = model(image)
            image = image * (normalisation_factor ** -1)
            image = image.squeeze(0).type(torch.uint8)
            for instance in output:
                op = instance["masks"]
                op = op >= 0.5
                image = draw_segmentation_masks(image=image, masks=op)

        # Draw segmentation mask on top of image
        masked_images.append(image)

    for masked_image in masked_images:
        show(masked_image)
    #grid = make_grid(masked_images)
    #show(grid)


if __name__ == "__main__":
    main()
