import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_segmentation_masks, make_grid
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as MaskRCNN
import argparse
import os
import numpy as np
import torchvision.transforms.functional as f
import matplotlib.pyplot as plt


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


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument("image_dir", help="path to directory containing images to run through the model")
    parser.add_argument("prediction_dir", help="path to directory to save predictions made by the model")
    parser.add_argument("-i", "--index", type=int)
    args = parser.parse_args()
    return args


def main():
    args = command_line_args()

    start, end = 0, args.index
    if args.index is not None:
        start = args.index - 1
        if args.index == 0:
            start, end = 0, 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model from checkpoint file
    checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(MaskRCNN(num_classes=2)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    normalisation_factor = 1 / 255

    model.eval()
    masked_images = []
    print("Making predictions ...\n")
    for filename in os.listdir(args.image_dir)[start: end]:
        # Get image and convert to required format
        img_path = os.path.join(args.image_dir, filename)
        image = read_image(img_path, mode=ImageReadMode.RGB) * normalisation_factor
        image = image.float().unsqueeze(dim=0)

        # Get mask prediction for model
        with torch.inference_mode():
            predictions = model(image)
            image = image * (normalisation_factor ** -1)
            image = image.squeeze(0).type(torch.uint8)

            instance_masks = predictions[0]['masks'] >= 0.5
            instance_masks = instance_masks.squeeze(dim=1)

            image = draw_segmentation_masks(image=image, masks=instance_masks)

        # Draw segmentation mask on top of image
        masked_images.append(image)

    print("Saving predictions ...\n")
    # Save the predictions to specified directory
    for i, prediction in enumerate(masked_images):
        prediction = f.to_pil_image(prediction)
        prediction.save(os.path.join(args.prediction_dir, f"prediction_{i}.png"))

    grid = make_grid(masked_images)
    show(grid)


if __name__ == "__main__":
    main()
