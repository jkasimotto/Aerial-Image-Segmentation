from torchmetrics.functional import jaccard_index, dice
from model_analyzer import ModelAnalyzer
from torchvision.models.segmentation import fcn_resnet101
from torch.utils.data import DataLoader
import torch.profiler
from torch import nn
from dataset import PlanesDataset
import numpy as np
from tqdm import tqdm
import time
import argparse
import wandb
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train(model, criterion, optimizer, train_loader, test_loader, num_classes, device, analyser, epochs=1,
          use_wandb=False):
    """
    Trains the model for the specified number of epochs and performs validation every epoch. Also updates
    the best saved model throughout training process.
    """
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    start = time.time()

    # Create data structures to record performance of the model
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []

    # Training loop
    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch + 1}")

        # Epoch training
        train_epoch_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)

        # Epoch validation
        val_epoch_loss, epoch_iou, epoch_dice = test(model, criterion, test_loader, device, num_classes)

        # Log results to Weights anf Biases
        if use_wandb:
            wandb.log({
                'train_loss': train_epoch_loss,
                "val_loss": val_epoch_loss,
                "mIoU": epoch_iou,
                "dice": epoch_dice,
            })

        # Save model performance values
        train_loss.append(train_epoch_loss)
        test_loss.append(val_epoch_loss)
        iou_acc.append(epoch_iou)
        dice_acc.append(epoch_dice)

        # Update best model saved throughout training
        analyser.save_best_model(val_epoch_loss, epoch_iou, epoch, model, optimizer, criterion)

        print(
            f"Epochs [{epoch + 1}/{epochs}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
        print("---\n")

    end = time.time()

    # Saving the loss and accuracy plot after training is complete
    analyser.save_loss_plot(train_loss, test_loss)
    analyser.save_acc_plot(iou_acc, dice_acc)

    print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    """
    Trains the model for one epoch, iterating through all batches of the datalaoder.
    :return: The average loss of the epoch
    """
    print('[EPOCH TRAINING]')
    model.train()
    running_loss = 0
    for batch, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.to(device), labels.to(device)
        prediction = model(images)['out']
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)


def test(model, criterion, dataloader, device, num_classes):
    """
    Performs validation on the current model. Calculates mIoU and dice score of the model.
    :return: tuple containing validation loss, mIoU accuracy and dice score
    """
    print("[VALIDATING]")
    ious, dice_scores = list(), list()
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            loss = criterion(prediction, labels)
            running_loss += loss.item()
            prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)
            labels = labels.argmax(dim=1)
            iou = jaccard_index(prediction, labels, num_classes=num_classes).item()
            dice_score = dice(prediction, labels, num_classes=num_classes, ignore_index=0).item()
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
    parser.add_argument("checkpoint_dir",
                        help="path to directory for model checkpoint to be saved")
    parser.add_argument("-r", "--run-name", default="fcn",
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
                        help="number of classes for semantic segmentation")
    parser.add_argument("-wandb", "--wandb",
                        help="use weights and biases to log run", action='store_true')
    args = parser.parse_args()
    return args


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


def my_collate_fn(batch):
    images, labels = [], []
    for img, mask in batch:
        images.append(img)
        labels.append(mask)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels


def main():
    args = command_line_args()

    print(f'Starting run: {args.run_name}\n')

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
    }

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

    train_transform, test_transform = augmentations()

    train_dataset = PlanesDataset(img_dir=img_dir, mask_dir=mask_dir,
                                  num_classes=HYPER_PARAMS['num_classes'], transforms=train_transform)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir,
                                 num_classes=HYPER_PARAMS['num_classes'], transforms=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=HYPER_PARAMS['batch_size'],
                              shuffle=True, num_workers=2, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=HYPER_PARAMS['batch_size'],
                             num_workers=2, collate_fn=my_collate_fn)

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(fcn_resnet101(num_classes=HYPER_PARAMS['num_classes']), device_ids=device_ids).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    if args.wandb:
        wandb.init(project="FCN", entity="usyd-04a", config=HYPER_PARAMS, dir="./wandb_data")
        wandb.watch(model, criterion=criterion)

    analyser = ModelAnalyzer(checkpoint_dir=args.checkpoint_dir, run_name=args.run_name)

    model = train(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  device=device,
                  analyser=analyser,
                  epochs=HYPER_PARAMS['epochs'],
                  num_classes=HYPER_PARAMS['num_classes'],
                  use_wandb=args.wandb)

    analyser.save_model(model=model,
                        epochs=HYPER_PARAMS['epochs'],
                        optimizer=optimizer,
                        criterion=criterion,
                        batch_size=HYPER_PARAMS['batch_size'],
                        lr=HYPER_PARAMS['learning_rate'])


if __name__ == "__main__":
    main()
