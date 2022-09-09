from torchvision.utils import draw_segmentation_masks, make_grid
import argparse
import numpy as np
import torchvision.transforms.functional as f
from torchvision import transforms
from torch.utils.data import DataLoader
from inference_dataset import InferenceDataset
from utils import *
from torch import nn


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


def inference(model, dataloader, device, image_idx):
    print("\n=================")
    print("| Show Predictions |")
    print("=================\n")

    masked_images = []
    model.eval()
    with torch.inference_mode():
        for idx, (normalised_image, images) in enumerate(dataloader):
            if idx >= image_idx[0] and idx < image_idx[1]:

                normalised_image = normalised_image.to(device)

                # Do Prediction
                prediction = model(normalised_image)['out']
                prediction = prediction.softmax(dim=1).argmax(dim=1) > 0

                # Squeeze Image
                image = images[0] * 255
                image = image.type(torch.uint8).cpu()

                image_with_mask = draw_segmentation_masks(image=image, masks=prediction, colors="red", alpha=0.5)
                masked_images.append(image_with_mask)

    grid = make_grid(masked_images)
    show(grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument("image_dir", help="path to directory containing images to run through the model")
    parser.add_argument("-s", "--start_index", type=int)
    parser.add_argument("-e", "--end_index", type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    image_idx = [args.start_index, args.end_index]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    inference_dataset = InferenceDataset(img_dir=args.image_dir, transform=transform)
    inference_loader = DataLoader(inference_dataset, batch_size=1)

    model = torch.load(args.model)
    model = nn.DataParallel(model).to(device)


    inference(model=model,
              dataloader=inference_loader,
              device=device,
              image_idx=image_idx)


if __name__ == "__main__":
    main()