import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import time
from tqdm import tqdm
from torchmetrics.functional import jaccard_index, dice

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import PlanesDataset
from model_analyzer import ModelAnalyzer

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as MaskRCNN

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, dataloader, device, print_every):
    print('[EPOCH TRAINING]')
    model.train()

    running_loss = 0
    for images, targets in tqdm(dataloader):
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

    return running_loss / len(dataloader)


def test_one_epoch(model, dataloader, device, num_classes):
    print('[EPOCH VALIDATING]')
    model.eval()

    ious, dice_scores = [], []
    with torch.inference_mode():
        for images, targets in tqdm(dataloader):
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
                    pred_mask_union = pred_mask_union.to(device).logical_or(mask)

                targ_seg_mask = target['seg_mask']
                # calculate iou and dice score
                iou = jaccard_index(pred_mask_union.to(device), targ_seg_mask.to(device), num_classes=num_classes).item()
                dice_score = dice(pred_mask_union.to(device), targ_seg_mask.to(device), num_classes=num_classes).item()
                ious.append(iou), dice_scores.append(dice_score)


    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return iou_acc, dice_acc

def augmentations():
    train_transforms = A.Compose([
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()])

    test_transforms = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()])

    return train_transforms, test_transforms

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="path to directory containing test and train images")
    parser.add_argument("checkpoint_dir",
                        help="path to directory for model checkpoint to be saved")
    parser.add_argument("-c", "--checkpoint", default="maskrcnn",
                        help="used for naming output files")
    parser.add_argument("-b", '--batch-size', default=16, type=int,
                        help="dataloader batch size")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,
                        help="learning rate to be applied to the model")
    parser.add_argument("-e", "--epochs", default=1, type=int,
                        help="number of epochs to train the model for")
    parser.add_argument("-w", "--workers", default=2, type=int,
                        help="number of workers used in the dataloader")
    parser.add_argument("-n", "--num-classes", default=2, type=int,
                        help="number of classes for instance segmentation")
    args = parser.parse_args()
    return args


def main():
    args = command_line_args()

    print(f'Starting run: {args.checkpoint}\n')

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

    train_transform, test_transform = augmentations()

    # Create object which loads input images and target masks and applies transform
    train_dataset = PlanesDataset(img_dir=train_img_dir, mask_dir=train_mask_dir,
                                  num_classes=HYPER_PARAMS['num_classes'], transforms=train_transform)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir, num_classes=HYPER_PARAMS['num_classes'],
                                 transforms=test_transform)


    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

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
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    # get the model
    model = MaskRCNN(
            weights=None,
            num_classes=HYPER_PARAMS['NUM_CLASSES'], # optional
            weights_backbone=None)
    # enable parallelism
    model = nn.DataParallel(model, device_ids=device_ids)
    # move model to the right device
    model.to(device)

    # get an optimizer, scheduler, and loss function
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=HYPER_PARAMS['LR'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # ----------------------
    # TRAINING
    # ----------------------

    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    start = time.time()

    analyzer = ModelAnalyzer(checkpoint_dir=args.checkpoint_dir, run_name=args.checkpoint)
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

        analyzer.save_best_model(epoch_iou, epoch, model, optimizer)

        print(
            f"Epochs [{epoch + 1}/{HYPER_PARAMS['EPOCHS']}], Avg Train Loss: {train_epoch_loss:.4f}")
        print("---\n")

    end = time.time()
    print(f"\nTraining took: {end - start:.2f}s")

    analyzer.save_loss_plot(train_loss)
    analyzer.save_acc_plot(iou_acc, dice_acc)
    analyzer.save_model(model, HYPER_PARAMS['EPOCHS'], optimizer, HYPER_PARAMS['BATCH_SIZE'], HYPER_PARAMS['LR'])


if __name__ == "__main__":
    main()
