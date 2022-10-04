from torchmetrics.functional import jaccard_index, dice
from torchvision.models.segmentation import deeplabv3_resnet101
from model_analyser import ModelAnalyzer
from torch.utils.data import DataLoader
from torch import nn
from dataset import PlanesDataset
import numpy as np
from tqdm import tqdm
import time
import argparse
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def train(model, criterion, optimizer, train_loader, test_loader, analyser, args, rank, num_gpus, use_wandb=False):
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    # Time how long it takes to train the model
    start = time.time()

    # Initialise arrays to store training results
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []

    # Training Loop
    for epoch in range(args.epochs):
        print(f"[INFO] Epoch {epoch + 1}")

        # Get training Loss
        train_epoch_loss = train_one_epoch(model, criterion, optimizer, train_loader, rank)
        # Get Validation Loss, mIoU and DICE for epoch
        val_epoch_loss, epoch_iou, epoch_dice = test(model, criterion, test_loader, rank, args)

        # Append data to array for graphing
        train_loss.append(train_epoch_loss)
        test_loss.append(val_epoch_loss)
        iou_acc.append(epoch_iou)
        dice_acc.append(epoch_dice)

        # Log results to Weights anf Biases
        if use_wandb:
            wandb.log({
                'train_loss': train_epoch_loss,
                "val_loss": val_epoch_loss,
                "mIoU": epoch_iou,
                "dice": epoch_dice,
            })

        # Update best model saved throughout training
        if rank == 0:
            analyser.save_best_model(val_epoch_loss, epoch_iou, epoch, model, optimizer, criterion)

        print(
            f"Epochs [{epoch + 1}/{args.epochs}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
        print("---\n")

    end = time.time()

    # Create plots for accuracy and loss
    if rank == 0:
        analyser.save_loss_plot(train_loss, test_loss)
        analyser.save_acc_plot(iou_acc, dice_acc)

    print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, criterion, optimizer, dataloader, rank):
    print('[EPOCH TRAINING]')

    # Set model in training mode
    model.train()
    running_loss = 0

    # Calculate loss per batch
    for batch, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.cuda(rank), labels.cuda(rank)

        prediction = model(images)['out']
        loss = criterion(prediction, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Return average loss for the epoch
    return running_loss / len(dataloader)


def test(model, criterion, dataloader, rank, args):
    print("[VALIDATING]")

    # Set model in evaluation mode
    model.eval()
    ious, dice_scores = list(), list()
    start = time.time()
    running_loss = 0

    # Calculate test loss, IoU and Dice coefficient accuracy measures
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.cuda(rank), labels.cuda(rank)

            prediction = model(images)['out']
            loss = criterion(prediction, labels)
            running_loss += loss.item()
            prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)  # (batch_size, w, h)

            labels = labels.argmax(dim=1)  # (batch_size, w, h)
            iou = jaccard_index(prediction, labels, num_classes=args.num_classes).item()
            dice_score = dice(prediction, labels, num_classes=args.num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)

    end = time.time()

    test_loss = running_loss / len(dataloader)
    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    # Return test loss, IoU and Dice coefficient for the epoch
    return test_loss, iou_acc, dice_acc


def command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir",
                        help="path to directory containing test and train images")
    parser.add_argument("checkpoint_dir",
                        help="directory for model checkpoint to be saved as")
    parser.add_argument("-r", '--run-name', default="deeplab",
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

def dist_train(rank, args, num_gpus):

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

    torch.distributed.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=num_gpus
    )
    torch.manual_seed(0)


    torch.cuda.set_device(rank)

    # ----------------------
    # CREATE DATASET
    # ----------------------

    # Get train and test directory paths
    train_img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    train_mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

    train_transform, test_transform = augmentations()

    # Create object which loads input images and target masks and applies transform
    train_dataset = PlanesDataset(img_dir=train_img_dir, mask_dir=train_mask_dir,
                                  num_classes=HYPER_PARAMS['num_classes'], transforms=train_transform)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir, num_classes=HYPER_PARAMS['num_classes'],
                                 transforms=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_gpus, rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=num_gpus, rank=rank
    )

    # Pass dataset to dataloader with predefined arguments
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(HYPER_PARAMS['batch_size'] / num_gpus),
        shuffle=False,
        num_workers=HYPER_PARAMS['num_workers'],
        drop_last=True,
        collate_fn=my_collate_fn,
        pin_memory=True,
        sampler=train_sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(HYPER_PARAMS['batch_size'] / num_gpus),
        num_workers=HYPER_PARAMS['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True,
        sampler=test_sampler
    )

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    # model = DDP(deeplabv3_resnet101(num_classes=HYPER_PARAMS['num_classes']), device_ids=device_ids, output_device=len(device_ids))
    model = deeplabv3_resnet101(num_classes=HYPER_PARAMS['num_classes']).cuda(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    if args.wandb:
        wandb.init(project="DeepLabV3", entity="usyd-04a", config=HYPER_PARAMS, dir="./wandb_data")
        wandb.watch(model, criterion=criterion)

    analyser = ModelAnalyzer(checkpoint_dir=args.checkpoint_dir, run_name=args.run_name)

    # Train model
    model = train(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  analyser=analyser,
                  args=args,
                  rank=rank,
                  num_gpus=num_gpus,
                  use_wandb=args.wandb)

    if rank == 0:
        # Save model after training
        analyser.save_model(model=model,
                            epochs=HYPER_PARAMS['epochs'],
                            optimizer=optimizer,
                            criterion=criterion,
                            batch_size=HYPER_PARAMS['batch_size'],
                            lr=HYPER_PARAMS['learning_rate'])

    dist.destroy_process_group()


def main():
    # Load in command line arguments
    args = command_line_args()

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)
    num_gpus = len(device_ids)

    # model = nn.DataParallel(deeplabv3_resnet101(num_classes=HYPER_PARAMS['num_classes']), device_ids=device_ids)
    # model.to(device)

    # OS Setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(num_gpus)
    mp.spawn(dist_train, nprocs=num_gpus, args=(args, num_gpus))



if __name__ == "__main__":
    main()
