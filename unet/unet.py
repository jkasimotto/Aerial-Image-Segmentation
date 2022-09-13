import argparse
import os
import time

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from albumentations.pytorch import ToTensorV2
from timm.models.layers import get_attn
from torch.utils.data import DataLoader
from torchmetrics.functional import dice, jaccard_index
from tqdm import tqdm

from model import UNET
from utils import (SaveBestModel, get_loaders, save_acc_plot, save_loss_plot,
                   save_model_2)


def train(model, criterion, optimizer, scheduler, train_loader, test_loader, num_classes, device, epochs=1,
          print_every=10):
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    start = time.time()

    save_best_model = SaveBestModel()
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []
    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch + 1}")

        train_epoch_loss = train_one_epoch(
            model, criterion, optimizer, train_loader, device, print_every)
        val_epoch_loss, epoch_iou, epoch_dice = test(
            model, criterion, test_loader, device, num_classes)
        scheduler.step()

        train_loss.append(train_epoch_loss)
        test_loss.append(val_epoch_loss)
        iou_acc.append(epoch_iou)
        dice_acc.append(epoch_dice)

        # if use_wandb:
        #     wandb.log({
        #         'epoch loss': train_epoch_loss,
        #         "test loss": val_epoch_loss,
        #         "epoch iou": epoch_iou,
        #         "epoch dice": epoch_dice,
        #     })

        save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)

        print(
            f"Epochs [{epoch + 1}/{epochs}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
        print("---\n")

    end = time.time()

    save_loss_plot(train_loss, test_loss, 'unet_loss.png')
    save_acc_plot(iou_acc, dice_acc, 'unet_accuracy.png')

    print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, criterion, optimizer, dataloader, device, print_every):
    print('[EPOCH TRAINING]')
    model.train()
    running_loss = 0
    for batch, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.to(device), labels.to(device)
        prediction = model(images).squeeze(dim=1)
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def test(model, criterion, dataloader, device, num_classes):
    print("[VALIDATING]")
    ious, dice_scores = list(), list()
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            # UNET outputs a single channel. Squeeze to match labels.
            prediction = model(images).squeeze(dim=1)
            loss = criterion(prediction, labels)
            running_loss += loss.item()
            prediction = torch.sigmoid(prediction) > 0.5
            iou = jaccard_index(prediction, labels.int(),
                                num_classes=num_classes).item()
            dice_score = dice(prediction, labels.int(),
                              num_classes=num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)
    test_loss = running_loss / len(dataloader)
    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return test_loss, iou_acc, dice_acc


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
    parser.add_argument("-u" ,"--use-wandb", default=False, type=bool,
                        help="Whether to log on wandb")
    args = parser.parse_args()
    return args


def main():
    args = command_line_args()

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'NUM_CLASSES': args.num_classes,
        'BATCH_SIZE': args.batch_size,
        'NUM_WORKERS': args.workers,
        'LR': args.learning_rate,
        'EPOCHS': args.epochs,
        'PIN_MEMORY': True
    }

    # if args.use_wandb:
    #     wandb.config = HYPER_PARAMS
    #     wandb.init(project="UNET", entity="usyd-04a",
    #                config=wandb.config, dir="./wandb_data")

    # ----------------------
    # CREATE DATASET
    # ----------------------
    img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

    # Augmentations to training set
    train_transforms = A.Compose([
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    # Augmentations to test set
    test_transforms = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    train_loader, test_loader = get_loaders(
        img_dir,
        mask_dir,
        test_img_dir,
        test_mask_dir,
        HYPER_PARAMS['BATCH_SIZE'],
        train_transforms,
        test_transforms,
        HYPER_PARAMS['NUM_WORKERS'],
        HYPER_PARAMS['PIN_MEMORY']
    )

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(
        UNET(in_channels=3, out_channels=1, attn=get_attn('ese')), device_ids=device_ids).to(device)
    criterion = nn.BCEWithLogitsLoss()  # binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=HYPER_PARAMS["LR"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # if args.use_wandb:
    #     wandb.watch(model, criterion=criterion)

    model = train(model,
                  criterion=criterion,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  scheduler=scheduler,
                  device=device,
                  epochs=HYPER_PARAMS["EPOCHS"],
                  print_every=30,
                  num_classes=HYPER_PARAMS['NUM_CLASSES'])

    save_model_2(model=model,
                 epochs=HYPER_PARAMS['EPOCHS'],
                 optimizer=optimizer,
                 criterion=criterion,
                 batch_size=HYPER_PARAMS['BATCH_SIZE'],
                 lr=HYPER_PARAMS['LR'],
                 filename='unet_final.pth')


# Do this so on Windows there are no issues when using NUM_WORKERS
if __name__ == "__main__":
    main()