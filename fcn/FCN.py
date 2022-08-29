from torchmetrics.functional import jaccard_index, dice
from utils import *
from torchvision.models.segmentation import fcn_resnet101
from torch.utils.data import DataLoader
from torch import nn
from dataset import PlanesDataset
import numpy as np
from tqdm import tqdm
import time
import argparse


def train(model, criterion, optimizer, dataloader, device, epochs=1, print_every=10):
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    avg_loss_list = []
    model.train()
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        print(f"Epochs [{epoch + 1}/{epochs}], Avg Loss: {running_loss / len(dataloader):.4f}")
        avg_loss_list.append(running_loss / len(dataloader))
    end = time.time()

    print(f"\nTraining took: {end - start:.2f}s")
    # plot_loss(avg_loss_list)

    return model


def test(model, dataloader, device, num_classes):
    print("\n=================")
    print("| Testing Model |")
    print("=================\n")

    ious, dice_scores = list(), list()
    model.eval()
    start = time.time()
    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)  # (batch_size, w, h)
            labels = labels.argmax(dim=1)  # (batch_size, w, h)
            iou = jaccard_index(prediction, labels, num_classes=num_classes).item()
            dice_score = dice(prediction, labels, num_classes=num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)
    end = time.time()

    print(f"\nTesting took: {end - start:.2f}s")

    print("\n=================")
    print("| Model Results |")
    print("-----------------")
    print(f'| mIoU: {np.mean(ious) * 100:.3f}%  |')
    print(f'| dice: {np.mean(dice_scores) * 100:.3f}%  |')
    print("=================\n")


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="path to directory containing test and train images")
    parser.add_argument("-c", "--checkpoint",
                        help="filename for model checkpoint to be saved as")
    parser.add_argument("-b", '--batch-size', default=16, type=int,
                        help="dataloader batch size")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,
                        help="learning rate to be applied to the model")
    parser.add_argument("-e", "--epochs", default=1, type=int,
                        help="number of epochs to train the model for")
    parser.add_argument("-w", "--workers", default=2, type=int,
                        help="number of workers used in the dataloader")
    parser.add_argument("-n", "--num-classes", default=2, type=int,
                        help="number of classes for semantic segmentation")
    args = parser.parse_args()
    return args


def main():
    args = command_line_args()

    # Needed to download model from internet
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'NUM_CLASSES': args.num_classes,
        'BATCH_SIZE': args.batch_size,
        'NUM_WORKERS': args.workers,
        'LR': args.learning_rate,
        'EPOCHS': args.epochs,
    }

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

    train_dataset = PlanesDataset(img_dir=img_dir, mask_dir=mask_dir)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], num_workers=2)

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(fcn_resnet101(num_classes=HYPER_PARAMS['NUM_CLASSES']), device_ids=device_ids).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['LR'])
    loss_fn = nn.BCEWithLogitsLoss()

    train(model=model,
          criterion=loss_fn,
          optimizer=optimizer,
          dataloader=train_loader,
          device=device,
          epochs=HYPER_PARAMS['EPOCHS'],
          print_every=30)

    save_model(model, args.checkpoint)

    test(model=model,
         dataloader=test_loader,
         device=device,
         num_classes=HYPER_PARAMS['NUM_CLASSES'])


if __name__ == "__main__":
    main()
