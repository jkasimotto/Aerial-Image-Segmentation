import torch
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from torchvision.utils import draw_segmentation_masks
from torchvision.models.segmentation import fcn_resnet101
import argparse
import os
import numpy as np
import torchvision.transforms.functional as f
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument("image_dir", help="path to directory containing images to run through the model")
    parser.add_argument("prediction_dir", help="path to directory to save predictions made by the model")
    parser.add_argument("-i", "--index", type=int)
    args = parser.parse_args()
    return args


def save_predictions(masked_images, dir_path):
    print("Saving predictions ...\n")
    # Save the predictions to specified directory
    for i, prediction in enumerate(masked_images):
        prediction = f.to_pil_image(prediction)
        prediction.save(os.path.join(dir_path, f"prediction_{i}.png"))


def augmentations():
    return A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()])


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
    model = nn.DataParallel(fcn_resnet101(num_classes=2)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    masked_images = []
    print("Making predictions ...\n")
    for filename in os.listdir(args.image_dir)[start: end]:
        # Get image and convert to required format
        img_path = os.path.join(args.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        original_image = read_image(img_path, mode=ImageReadMode.RGB)  # used for displaying final prediction

        transforms = augmentations()

        image = transforms(image=np.array(image))['image']
        image = image.float().unsqueeze(0)

        # Get mask prediction for model
        with torch.inference_mode():
            output = model(image)['out']
            output = output.softmax(dim=1).argmax(dim=1) > 0

        image_with_mask = draw_segmentation_masks(image=original_image, masks=output, colors="red", alpha=0.5)
        masked_images.append(image_with_mask)

    save_predictions(masked_images, args.prediction_dir)


if __name__ == "__main__":
    main()
