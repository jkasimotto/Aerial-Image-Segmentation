import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from tqdm import tqdm

from dataset import PlanesDataset
from utils import SaveBestModel
from utils import collate_fn

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as MaskRCNN


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="path to directory containing test and train images")
    parser.add_argument("checkpoint",
                        help="path to directory for model checkpoint to be saved")
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


# needs to be updated
def train_one_epoch(model, optimizer, dataloader, device, print_every):
    print('[EPOCH TRAINING]')
    model.train()

    running_loss = 0
    for batch, (images, targets) in enumerate(tqdm(dataloader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        #if (batch + 1) % print_every == 0:
        #    print(f"Step [{batch + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return loss_value / len(dataloader)


# have not touched yet, needs a complete re-work
def test(model, criterion, dataloader, device, num_classes):
    print("[VALIDATING]")
    ious, dice_scores = [], []
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        for images, targets in tqdm(dataloader):
            images, targets = images.to(device), targets.to(device)
            prediction = model(images)['out']
            loss = criterion(prediction, targets)
            running_loss += loss.item()
            prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)  # (batch_size, w, h)
            targets = targets.argmax(dim=1)  # (batch_size, w, h)
            iou = jaccard_index(prediction, targets, num_classes=num_classes).item()
            dice_score = dice(prediction, targets, num_classes=num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)

    test_loss = running_loss / len(dataloader)
    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return test_loss, iou_acc, dice_acc


def main():
    args = command_line_args()

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

    img_dir = os.path.join(args.data_dir, 'train_small/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train_small/greyscale_masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'train_small/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'train_small/greyscale_masks_tiled')

    train_dataset = PlanesDataset(img_dir, mask_dir)
    test_dataset = PlanesDataset(test_img_dir, test_mask_dir)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=HYPER_PARAMS['BATCH_SIZE'],
        shuffle=True,
        num_workers=HYPER_PARAMS['NUM_WORKERS'],
        collate_fn=collate_fn)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=HYPER_PARAMS['BATCH_SIZE'],
        shuffle=False,
        num_workers=HYPER_PARAMS['NUM_WORKERS'],
        collate_fn=collate_fn)

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model
    model = MaskRCNN(
            weights=None,
            num_classes=HYPER_PARAMS['NUM_CLASSES'], # optional
            weights_backbone=None)
    # enable parallelism
    model = nn.DataParallel(model)
    # move model to the right device
    model.to(device)

    # get an optimizer, scheduler, and loss function
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=HYPER_PARAMS['LR'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # ----------------------
    # TRAINING
    # ----------------------

    save_best_model = SaveBestModel(args.checkpoint)
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []
    for epoch in range(HYPER_PARAMS['EPOCHS']):
        print(f"[INFO] Epoch {epoch + 1}")

        # train for one epoch
        train_epoch_loss = train_one_epoch(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                device=device,
                print_every=1)

        # calculate loss, iou and dice for the epoch
        val_epoch_loss, epoch_iou, epoch_dice = test(
                model=model,
                criterion=nn.BCEWithLogitsLoss(),
                dataloader=test_loader,
                device=device,
                num_classes=HYPER_PARAMS['NUM_CLASSES'])

        # update the learning rate
        scheduler.step()

        train_loss.append(train_epoch_loss)
        test_loss.append(val_epoch_loss)
        iou_acc.append(epoch_iou)
        dice_acc.append(epoch_dice)

        save_best_model(val_epoch_loss, epoch_iou, epoch, model, optimizer, criterion)

        print(
            f"Epochs [{epoch + 1}/{epochs}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
        print("---\n")

    save_loss_plot(train_loss, test_loss, os.path.join(checkpoint_dir, 'fcn_loss.png'))
    save_acc_plot(iou_acc, dice_acc, os.path.join(checkpoint_dir, 'fcn_accuracy.png'))

    """
    # make the plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        loss_plot, color='red', linestyle='-',
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot')
    print("That's it!")
    """
    

if __name__ == "__main__":
    main()
