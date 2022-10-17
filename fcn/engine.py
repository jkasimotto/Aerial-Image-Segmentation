from utils import is_main_node, get_device, get_memory_format
from torch.cuda.amp import autocast
from tqdm import tqdm
import torch
from torchmetrics.functional import jaccard_index, dice
import numpy as np
import time
import wandb


def train(model, criterion, optimizer, train_loader, test_loader, analyser, args, scaler=None, rank=None):
    """
    Trains the model for the specified number of epochs and performs validation every epoch. Also updates
    the best saved model throughout training process.
    """
    if is_main_node(rank):
        print("\n==================")
        print("| Training Model |")
        print("==================\n")

    assert args.get('amp').get('enabled') == (scaler is not None), "Scaler should be not None if AMP is enabled"

    start = time.time()

    # Create data structures to record performance of the model
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []

    # Training loop
    for epoch in range(args.get('hyper-params').get('epochs')):
        if is_main_node(rank):
            print(f"[INFO] Epoch {epoch + 1}")

        # Epoch training
        train_epoch_loss = train_one_epoch(model, criterion, optimizer, scaler, train_loader, args, rank)

        # Epoch validation
        val_epoch_loss, epoch_iou, epoch_dice = test(model, criterion, test_loader, args, rank)

        # Log results to Weights and Biases
        if args.get('wandb').get('enabled') and is_main_node(rank):
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
        if is_main_node(rank):
            analyser.save_best_model(val_epoch_loss, epoch_iou, epoch, model, optimizer, criterion)

            print(
                f"Epochs [{epoch + 1}/{args.get('hyper-params').get('epochs')}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
            print("---\n")

    end = time.time()

    # Saving the loss and accuracy plot after training is complete
    if is_main_node(rank):
        analyser.save_loss_plot(train_loss, test_loss)
        analyser.save_acc_plot(iou_acc, dice_acc)

        print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, criterion, optimizer, scaler, dataloader, args, rank):
    """
    Trains the model for one epoch, iterating through all batches of the datalaoder.
    :return: The average loss of the epoch
    """
    if is_main_node(rank):
        print('[EPOCH TRAINING]')

    running_loss = 0
    use_amp = args.get('amp').get('enabled')
    device = get_device(args)

    model.train()
    for batch, (images, labels) in enumerate(tqdm(dataloader, disable=not is_main_node(rank))):
        images = images.to(device, memory_format=get_memory_format(args))
        labels = labels.to(device, memory_format=get_memory_format(args))

        with autocast(enabled=use_amp):
            prediction = model(images)['out']
            loss = criterion(prediction, labels)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def test(model, criterion, dataloader, args, rank):
    """
    Performs validation on the current model. Calculates mIoU and dice score of the model.
    :return: tuple containing validation loss, mIoU accuracy and dice score
    """
    if is_main_node(rank):
        print("[VALIDATING]")

    running_loss = 0
    ious, dice_scores = list(), list()
    use_amp = args.get('amp').get('enabled')
    num_classes = args.get('config').get('classes')
    device = get_device(args)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=not is_main_node(rank)):
            images = images.to(device, memory_format=get_memory_format(args))
            labels = labels.to(device, memory_format=get_memory_format(args))

            with autocast(enabled=use_amp):
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

    if is_main_node(rank):
        print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return test_loss, iou_acc, dice_acc
