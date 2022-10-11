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

# from utils import (SaveBestModel, get_loaders, save_acc_plot, save_loss_plot,
#                    save_model_2)


def train(model, criterion, optimizer, train_loader, test_loader, analyser, args, scaler=None, rank=None):
    """
    Trains the model for the specified number of epochs and performs validation every epoch. Also updates
    the best saved model throughout training process.
    """
    if is_main_node(rank):
        print("\n==================")
        print("| Training Model |")
        print("==================\n")

    assert args.get('config').get('amp') == (scaler is not None), "Scaler should be not None if AMP is enabled"

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
    use_amp = args.get('config').get('amp')
    device = get_device(args)

    model.train()
    for batch, (images, labels) in enumerate(tqdm(dataloader, disable=not is_main_node(rank))):
        images = images.to(device, memory_format=get_memory_format(args))
        labels = labels.to(device, memory_format=get_memory_format(args))

        with autocast(enabled=use_amp):
            prediction = model(images).squeeze(dim=1)
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

    # print('[EPOCH TRAINING]')
    # model.train()
    # running_loss = 0
    # for batch, (images, labels) in enumerate(tqdm(dataloader)):
    #     images, labels = images.cuda(rank), labels.cuda(rank)
    #     prediction = model(images).squeeze(dim=1)
    #     loss = criterion(prediction, labels)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()
    # return running_loss / len(dataloader)


def test(model, criterion, dataloader, args, rank):
    """
    Performs validation on the current model. Calculates mIoU and dice score of the model.
    :return: tuple containing validation loss, mIoU accuracy and dice score
    """
    if is_main_node(rank):
        print("[VALIDATING]")

    running_loss = 0
    ious, dice_scores = list(), list()
    use_amp = args.get('config').get('amp')
    num_classes = args.get('config').get('classes')
    device = get_device(args)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=not is_main_node(rank)):
            images = images.to(device, memory_format=get_memory_format(args))
            labels = labels.to(device, memory_format=get_memory_format(args))

            with autocast(enabled=use_amp):
                prediction = model(images).squeeze(dim=1)
                loss = criterion(prediction, labels)

            running_loss += loss.item()
            prediction = torch.sigmoid(prediction) > 0.5
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

    # print("[VALIDATING]")
    # ious, dice_scores = list(), list()
    # model.eval()
    # running_loss = 0
    # with torch.inference_mode():
    #     for images, labels in tqdm(dataloader):
    #         images, labels = images.cuda(rank), labels.cuda(rank)
    #         # UNET outputs a single channel. Squeeze to match labels.
    #         prediction = model(images).squeeze(dim=1)
    #         loss = criterion(prediction, labels)
    #         running_loss += loss.item()
    #         prediction = torch.sigmoid(prediction) > 0.5
    #         iou = jaccard_index(prediction, labels.int(),
    #                             num_classes=num_classes).item()
    #         dice_score = dice(prediction, labels.int(),
    #                           num_classes=num_classes, ignore_index=0).item()
    #         ious.append(iou), dice_scores.append(dice_score)
    # test_loss = running_loss / len(dataloader)
    # iou_acc = np.mean(ious)
    # dice_acc = np.mean(dice_scores)

    # print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    # return test_loss, iou_acc, dice_acc


# def command_line_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("data_dir",
#                         help="path to directory containing test and train images")
#     parser.add_argument("-c", "--checkpoint",
#                         help="filename for model checkpoint to be saved as")
#     parser.add_argument("-b", '--batch-size', default=16, type=int,
#                         help="dataloader batch size")
#     parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,
#                         help="learning rate to be applied to the model")
#     parser.add_argument("-e", "--epochs", default=1, type=int,
#                         help="number of epochs to train the model for")
#     parser.add_argument("-w", "--workers", default=2, type=int,
#                         help="number of workers used in the dataloader")
#     parser.add_argument("-n", "--num-classes", default=2, type=int,
#                         help="number of classes for semantic segmentation")
#     parser.add_argument("-u" ,"--use-wandb", default=False, type=bool,
#                         help="Whether to log on wandb")
#     args = parser.parse_args()
#     return args

def training_setup(gpu, args):
    torch.manual_seed(0)

    # Setup process if distributed is enabled
    rank = None
    if args.get('config').get('distributed'):
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
    if args.get('config').get('amp'):
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
    if args.get('config').get('distributed'):
        dist.destroy_process_group()

# def training_setup(gpu, args):
    
#     # ----------------------
#     # DEFINE HYPER PARAMETERS
#     # ----------------------

#     HYPER_PARAMS = {
#         'NUM_CLASSES': args.num_classes,
#         'BATCH_SIZE': args.batch_size,
#         'NUM_WORKERS': args.workers,
#         'LR': args.learning_rate,
#         'EPOCHS': args.epochs,
#         'PIN_MEMORY': True
#     }

#     torch.distributed.init_process_group(
#         backend='nccl',
#         rank=rank,
#         world_size=num_gpus
#     )
#     torch.manual_seed(0)

#     torch.cuda.set_device(rank)

#     # ----------------------
#     # CREATE DATASET
#     # ----------------------
#     img_dir = os.path.join(args.data_dir, 'train/images_tiled')
#     mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
#     test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
#     test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

#     #Augmentations to training and testing set
#     train_transforms, test_transforms = augmentations()

#     train_loader, test_loader = get_loaders(
#         img_dir,
#         mask_dir,
#         test_img_dir,
#         test_mask_dir,
#         HYPER_PARAMS['BATCH_SIZE'],
#         train_transforms,
#         test_transforms,
#         num_gpus,
#         rank,
#         HYPER_PARAMS['NUM_WORKERS'],
#         HYPER_PARAMS['PIN_MEMORY']
#     )

#     # ----------------------
#     # DEFINE MODEL
#     # ----------------------

#     # device_ids = [i for i in range(torch.cuda.device_count())]
#     device_ids=[rank]
#     model = nn.DataParallel(UNET(in_channels=3, out_channels=1), device_ids=device_ids).cuda(rank)
#     model = DDP(model, device_ids=[rank])
#     criterion = nn.BCEWithLogitsLoss()  # binary cross entropy loss
#     optimizer = optim.Adam(model.parameters(), lr=HYPER_PARAMS["LR"])
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#     if args.use_wandb:
#         wandb.config = HYPER_PARAMS
#         wandb.init(project="UNET", entity="usyd-04a",
#                    config=wandb.config, dir="./wandb_data")

#     model = train(model,
#                   criterion=criterion,
#                   optimizer=optimizer,
#                   scheduler=scheduler,
#                   train_loader=train_loader,
#                   test_loader=test_loader,
#                   rank=rank,
#                   num_gpus=num_gpus,
#                   epochs=HYPER_PARAMS["EPOCHS"],
#                   print_every=30,
#                   num_classes=HYPER_PARAMS['NUM_CLASSES'],
#                   use_wandb=args.use_wandb)

#     if rank == 0:
#         save_model_2(model=model,
#                     epochs=HYPER_PARAMS['EPOCHS'],
#                     optimizer=optimizer,
#                     criterion=criterion,
#                     batch_size=HYPER_PARAMS['BATCH_SIZE'],
#                     lr=HYPER_PARAMS['LR'],
#                     filename='unet_final.pth')
    
#     dist.destroy_process_group()


def main():
    args = read_config_file()

    print(f'Starting run: {args.get("config").get("run")}\n')
    print(f'AMP: {args.get("config").get("amp")}')
    print(f'Channels Last: {args.get("config").get("channels-last")}')
    print(f'Distributed: {args.get("config").get("distributed")}')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    torch.cuda.empty_cache()

    if args.get('config').get('distributed'):
        dist_env_setup(args)
        mp.spawn(training_setup, nprocs=args.get("distributed").get('ngpus'), args=(args,))
    else:
        training_setup(None, args)


# Do this so on Windows there are no issues when using NUM_WORKERS
if __name__ == "__main__":
    main()
