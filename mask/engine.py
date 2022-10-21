import contextlib
import math
import os
import sys
import time
import torch
import utils
import wandb
from tqdm import tqdm
from utils import is_main_node, get_device
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from logger import MetricLogger


def train(model, optimizer, train_loader, test_loader, analyser, args, scaler=None, rank=None):
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
        train_epoch_loss = train_one_epoch(model, optimizer, scaler, train_loader, args, rank)

        # Epoch validation
        epoch_iou = evaluate(model, test_loader, args, rank)

        # Log results to Weights and Biases
        if args.get('wandb').get('enabled') and is_main_node(rank):
            wandb.log({
                'train_loss': train_epoch_loss,
                "mIoU": epoch_iou,
            })

        # Save model performance values
        train_loss.append(train_epoch_loss)
        iou_acc.append(epoch_iou)

        # Update best model saved throughout training
        if is_main_node(rank):
            analyser.save_best_model(epoch_iou, epoch, model, optimizer)

            print(
                f"Epochs [{epoch + 1}/{args.get('hyper-params').get('epochs')}], Avg Train Loss: {train_epoch_loss:.4f}")
            print("---\n")

    end = time.time()

    # Saving the loss and accuracy plot after training is complete
    if is_main_node(rank):
        analyser.save_loss_plot(train_loss, test_loss)
        analyser.save_acc_plot(iou_acc, dice_acc)

        print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, optimizer, scaler, dataloader, args, rank):
    model.train()

    running_loss = 0
    use_amp = args.get('amp').get('enabled')
    device = get_device(args)

    for batch, (images, targets) in enumerate(tqdm(dataloader, disable=not is_main_node(rank))):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        running_loss += loss_value

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

    return running_loss / len(dataloader)


@torch.inference_mode()
def evaluate(model, data_loader, args, rank):
    device = get_device(args)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = ["segm"]
        coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in tqdm(data_loader, disable=not is_main_node(rank)):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    iou = coco_evaluator.coco_eval['segm'].stats[0]

    if is_main_node(rank):
        print(f"Accuracy: mIoU= {iou * 100:.3f}%")

    return iou
