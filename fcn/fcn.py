from torchmetrics.functional import jaccard_index, dice
from model_analyser import ModelAnalyser
from torchvision.models.segmentation import fcn_resnet101
from torch.utils.data import DataLoader
import torch.profiler
from torch import nn
from dataset import PlanesDataset
import numpy as np
from tqdm import tqdm
import time
import wandb
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import augmentations, my_collate_fn, command_line_args


def train(model, criterion, optimizer, train_loader, test_loader, analyser, args, rank):
    """
    Trains the model for the specified number of epochs and performs validation every epoch. Also updates
    the best saved model throughout training process.
    """
    if rank == 0:
        print("\n==================")
        print("| Training Model |")
        print("==================\n")

    start = time.time()

    # Create data structures to record performance of the model
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []

    # Training loop
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"[INFO] Epoch {epoch + 1}")

        # Epoch training
        train_epoch_loss = train_one_epoch(model, criterion, optimizer, train_loader, args, rank)

        # Epoch validation
        val_epoch_loss, epoch_iou, epoch_dice = test(model, criterion, test_loader, args, rank)

        # Log results to Weights anf Biases
        if args.wandb:
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
        if rank == 0:
            analyser.save_best_model(val_epoch_loss, epoch_iou, epoch, model, optimizer, criterion)

            print(
                f"Epochs [{epoch + 1}/{args.epochs}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
            print("---\n")

    end = time.time()

    # Saving the loss and accuracy plot after training is complete
    if rank == 0:
        analyser.save_loss_plot(train_loss, test_loss)
        analyser.save_acc_plot(iou_acc, dice_acc)

        print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, criterion, optimizer, dataloader, args, rank):
    """
    Trains the model for one epoch, iterating through all batches of the datalaoder.
    :return: The average loss of the epoch
    """
    if rank == 0:
        print('[EPOCH TRAINING]')
    model.train()
    running_loss = 0
    for batch, (images, labels) in enumerate(tqdm(dataloader, disable=rank != 0)):
        images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
        # with torch.autocast('cuda'):
        prediction = model(images)['out']
        loss = criterion(prediction, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)


def test(model, criterion, dataloader, args, rank):
    """
    Performs validation on the current model. Calculates mIoU and dice score of the model.
    :return: tuple containing validation loss, mIoU accuracy and dice score
    """
    if rank == 0:
        print("[VALIDATING]")
    ious, dice_scores = list(), list()
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=rank != 0):
            images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
            # with torch.autocast('cuda'):
            prediction = model(images)['out']
            loss = criterion(prediction, labels)
            running_loss += loss.item()
            prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)
            labels = labels.argmax(dim=1)
            iou = jaccard_index(prediction, labels, num_classes=args.num_classes).item()
            dice_score = dice(prediction, labels, num_classes=args.num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)

    test_loss = running_loss / len(dataloader)
    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    if rank == 0:
        print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return test_loss, iou_acc, dice_acc


def dist_train(gpu, args):
    args.gpu = gpu
    rank = args.local_ranks * args.ngpus + gpu

    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank,
    )
    torch.manual_seed(0)

    torch.cuda.set_device(args.gpu)

    # Setup dataset, augmentations, datalaoders
    img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

    train_transform, test_transform = augmentations()

    train_dataset = PlanesDataset(
        img_dir=img_dir, mask_dir=mask_dir,
        num_classes=args.num_classes, transforms=train_transform,
    )
    test_dataset = PlanesDataset(
        img_dir=test_img_dir, mask_dir=test_mask_dir,
        num_classes=args.num_classes, transforms=test_transform,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size / args.ngpus),
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        collate_fn=my_collate_fn,
        pin_memory=True,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size / args.ngpus),
        num_workers=args.workers,
        collate_fn=my_collate_fn,
        pin_memory=True,
        sampler=test_sampler,
    )

    # Setup Model, optimiser and criterion
    model = fcn_resnet101(num_classes=args.num_classes).cuda(args.gpu)
    model = DDP(model, device_ids=[args.gpu])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Setup analyser for model checkpoint saving
    analyser = ModelAnalyser(checkpoint_dir=args.checkpoint_dir, run_name=args.run_name)

    # Training loop
    model = train(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  analyser=analyser,
                  args=args,
                  rank=rank)

    if rank == 0:
        analyser.save_model(model=model,
                            epochs=args.epochs,
                            optimizer=optimizer,
                            criterion=criterion,
                            batch_size=args.batch_size,
                            lr=args.learning_rate)

    dist.destroy_process_group()


def main():
    args = command_line_args()

    print(f'Starting run: {args.run_name}\n')

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    # DDP Setup
    args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
    args.world_size = args.nodes * args.ngpus
    os.environ['MASTER_ADDR'] = args.ip_address
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    mp.spawn(dist_train, nprocs=args.ngpus, args=(args,))


if __name__ == "__main__":
    main()
