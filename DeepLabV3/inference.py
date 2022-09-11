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


def inference(model, dataloader, device, prediction_dir):
    print("\n=================")
    print("| Show Predictions |")
    print("=================\n")

    masked_images = []
    model.eval()
    with torch.inference_mode():
        for idx, (normalised_image, images) in enumerate(dataloader):

            normalised_image = normalised_image.to(device)

            # Do Prediction
            prediction = model(normalised_image)['out']
            prediction = prediction.softmax(dim=1).argmax(dim=1) > 0

            # Squeeze Image
            image = images[0] * 255
            image = image.type(torch.uint8).cpu()

            image_with_mask = draw_segmentation_masks(image=image, masks=prediction, colors="red", alpha=0.7)
            masked_images.append(image_with_mask)

    print("Saving predictions ...\n")
    for idx, prediction in enumerate(masked_images):
        prediction = f.to_pil_image(prediction)
        os.makedirs(prediction_dir, exist_ok=True)
        prediction.save(os.path.join(prediction_dir, f"prediction_{idx}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="checkpoint file for pretrained model")
    parser.add_argument("image_dir", help="path to directory containing images to run through the model")
    parser.add_argument("prediction_dir", help="path to directory to save predictions made by the model")
    parser.add_argument("-s", "--start_index", type=int)
    parser.add_argument("-e", "--end_index", type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    inference_dataset = InferenceDataset(img_dir=args.image_dir, start_idx=args.start_index, end_idx=args.end_index, transform=transform)
    inference_loader = DataLoader(inference_dataset, batch_size=1)

    checkpoint = torch.load(args.model)
    model = nn.DataParallel(
        torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', num_classes=2)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])


    inference(model=model,
              dataloader=inference_loader,
              device=device,
              prediction_dir=args.prediction_dir)


if __name__ == "__main__":
    main()