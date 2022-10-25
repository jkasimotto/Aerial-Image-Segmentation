from torchmetrics.functional import jaccard_index, dice
from model_analyser import ModelAnalyser
import torch.profiler
from torch import nn
import numpy as np
from tqdm import tqdm
import time
import wandb
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from utils import (
    read_config_file,
    get_data_loaders,
    is_main_node,
    get_model,
    dist_env_setup,
    dist_process_setup,
    get_memory_format,
    get_device,
)


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

        # Log results to Weights anf Biases
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
        if args.get('channels-last').get('enabled'):
            labels = labels.unsqueeze(dim=1)
        labels = labels.to(device, memory_format=get_memory_format(args))

        with autocast(enabled=use_amp):
            if args.get('channels-last').get('enabled'):
                labels = labels.squeeze(dim=1)
            prediction = model(images).squeeze(dim=1)
            loss = criterion(prediction, labels)
            if args.get('channels-last').get('enabled'):
                labels = labels.unsqueeze(dim=1)

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
            if args.get('channels-last').get('enabled'):
                labels = labels.unsqueeze(dim=1)
            labels = labels.to(device, memory_format=get_memory_format(args))

            with autocast(enabled=use_amp):
                if args.get('channels-last').get('enabled'):
                    labels = labels.squeeze(dim=1)
                prediction = model(images).squeeze(dim=1)
                loss = criterion(prediction, labels)
                if args.get('channels-last').get('enabled'):
                    labels = labels.unsqueeze(dim=1)

            running_loss += loss.item()
            prediction = torch.sigmoid(prediction) > 0.5
            # labels = labels.argmax(dim=1)
            iou = jaccard_index(prediction, labels.int(), num_classes=num_classes).item()
            dice_score = dice(prediction, labels.int(), num_classes=num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)

    test_loss = running_loss / len(dataloader)
    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    if is_main_node(rank):
        print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return test_loss, iou_acc, dice_acc


def training_setup(gpu, args):
    torch.manual_seed(0)

    # Setup process if distributed is enabled
    rank = None
    if args.get('distributed').get('enabled'):
        rank = dist_process_setup(args, gpu)

    hyper_params = args.get('hyper-params')

    # Setup dataloaders, model, criterion and optimiser
    train_loader, test_loader = get_data_loaders(args, rank)
    model = get_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.get('learning-rate'))
    criterion = nn.BCEWithLogitsLoss()

    # Configure Weights and Biases
    if args.get('wandb').get('enabled') and is_main_node(rank):
        wandb.init(project=args.get('wandb').get('project-name'), entity="usyd-04a", config=args, dir="./wandb_data")
        wandb.watch(model, criterion=criterion)
        wandb.run.name = args.get('config').get('run')

    # Create analyser for saving model checkpoints
    analyser = ModelAnalyser(
        checkpoint_dir=args.get('config').get('checkpoint-dir'),
        run_name=args.get('config').get('run'),
    )

    # Setup Gradient Scaler if AMP is enabled
    scaler = None
    if args.get('amp').get('enabled'):
        scaler = GradScaler()

    # Training loop
    model = train(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  scaler=scaler,
                  analyser=analyser,
                  args=args,
                  rank=rank)

    if is_main_node(rank):
        analyser.save_model(model=model,
                            epochs=hyper_params.get('epochs'),
                            optimizer=optimizer,
                            criterion=criterion,
                            batch_size=hyper_params.get('batch-size'),
                            lr=hyper_params.get('learning-rate'))

    # Clean up distributed process
    if args.get('distributed').get('enabled'):
        dist.destroy_process_group()


def main():
    args = read_config_file()

    print(f'Starting run: {args.get("config").get("run")}\n')
    print(f'AMP: {args.get("amp").get("enabled")}')
    print(f'Channels Last: {args.get("channels-last").get("enabled")}')
    print(f'Distributed: {args.get("distributed").get("enabled")}')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    torch.cuda.empty_cache()

    if args.get('distributed').get('enabled'):
        dist_env_setup(args)
        mp.spawn(training_setup, nprocs=args.get("distributed").get('ngpus'), args=(args,))
    else:
        training_setup(None, args)


if __name__ == "__main__":
    main()
