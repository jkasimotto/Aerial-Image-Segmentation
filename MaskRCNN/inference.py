import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_segmentation_masks, make_grid
from torchvision.models.segmentation import fcn_resnet101
import argparse
import os
import numpy as np
import torchvision.transforms.functional as f
import matplotlib.pyplot as plt
from torch import nn
import os
import numpy as np
import torch
from PIL import Image
from dataset import PlanesDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T


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

    # model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
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
            output = model(image)['masks']
            output = output.softmax(dim=1).argmax(dim=1) > 0

        image = image * (normalisation_factor ** -1)

        # Draw segmentation mask on top of image
        image = image.squeeze(0).type(torch.uint8)
        image_with_mask = draw_segmentation_masks(image=image, masks=output, colors="red")
        masked_images.append(image_with_mask)

    grid = make_grid(masked_images)
    show(grid)


if __name__ == "__main__":
    main()
