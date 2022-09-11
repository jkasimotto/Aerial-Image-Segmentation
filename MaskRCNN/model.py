import matplotlib.pyplot as plt
import math
import numpy as np
import os
from tqdm import tqdm
from torchmetrics.functional import jaccard_index, dice

from dataset import PlanesDataset
from utils import *

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as MaskRCNN


def train_one_epoch(model, optimizer, dataloader, device, print_every):
    print('[EPOCH TRAINING]')
    model.train()

    running_loss = 0
    #for batch, (images, targets) in enumerate(tqdm(dataloader)):
    for batch, (images, targets) in enumerate(dataloader):
        # send the images and targets to the model
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # sum the classification and regression losses for
        # RPN and R-CNN, and the mask loss
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        running_loss += loss_value

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (batch + 1) % print_every == 0:
            print(f"Step [{batch + 1}/{len(dataloader)}] Loss: {loss_value:.4f}")

    return running_loss / len(dataloader)


def test_one_epoch(model, dataloader, device, num_classes):
    print('[EPOCH VALIDATING]')
    model.eval()

    ious, dice_scores = [], []
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(dataloader):
            # send the images and targets to the model
            images = list(image.to(device) for image in images)

            predictions = model(images)
            for prediction, target in zip(predictions, targets):
                # create an empty mask
                pred_mask_union = torch.zeros(512, 512, dtype=torch.uint8)
                # threshhold the prediction masks by probability >= 0.5
                binary_pred_masks = prediction['masks'] >= 0.5
                binary_pred_masks = binary_pred_masks.squeeze(dim=1)
                # union the prediction masks together
                for mask in binary_pred_masks:
                    pred_mask_union = pred_mask_union.logical_or(mask)

                targ_seg_mask = target['seg_mask']
                # calculate iou and dice score
                iou = jaccard_index(pred_mask_union, targ_seg_mask, num_classes=num_classes).item()
                dice_score = dice(pred_mask_union, targ_seg_mask, num_classes=num_classes).item()
                ious.append(iou), dice_scores.append(dice_score)


    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return iou_acc, dice_acc


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

    img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train/greyscale_masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/greyscale_masks_tiled')

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

        # validate the epoch
        epoch_iou, epoch_dice = test_one_epoch(
                model=model,
                dataloader=test_loader,
                device=device,
                num_classes=HYPER_PARAMS['NUM_CLASSES'])

        # update the learning rate
        scheduler.step()

        train_loss.append(train_epoch_loss)
        iou_acc.append(epoch_iou)
        dice_acc.append(epoch_dice)

        save_best_model(epoch_iou, epoch, model, optimizer)

        print(
            f"Epochs [{epoch + 1}/{HYPER_PARAMS['EPOCHS']}], Avg Train Loss: {train_epoch_loss:.4f}")
        print("---\n")

    save_loss_plot(train_loss, os.path.join(args.checkpoint, 'mask_loss.png'))
    save_acc_plot(iou_acc, dice_acc, os.path.join(args.checkpoint, 'mask_accuracy.png'))


if __name__ == "__main__":
    main()
