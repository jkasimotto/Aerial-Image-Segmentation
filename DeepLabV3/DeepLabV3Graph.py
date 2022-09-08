import torch
from torchmetrics.functional import jaccard_index, dice
from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from dataset import PlanesDataset
import numpy as np
from tqdm import tqdm
import time
import argparse
import wandb


def train(model, criterion, optimizer, train_loader, test_loader, num_classes, device, epochs=1, print_every=10):
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    wandb.watch(model, criterion=criterion)

    start = time.time()

    save_best_model = SaveBestModel()
    train_loss, test_loss = [], []
    iou_acc, dice_acc = [], []


    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch + 1}")

        # Get training Loss
        train_epoch_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, print_every)
        # Get Validation Loss, MIOU and DICE for epoch
        val_epoch_loss, epoch_iou, epoch_dice = test(model, criterion, test_loader, device, num_classes)

        # Append data to array for graphing
        train_loss.append(train_epoch_loss)
        test_loss.append(val_epoch_loss)
        iou_acc.append(epoch_iou)
        dice_acc.append(epoch_dice)

        wandb.log({
            'train_loss': train_epoch_loss,
            "val_loss": val_epoch_loss,
            "mIoU": epoch_iou,
            "dice": epoch_dice,
        })

        save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)

        print(
            f"Epochs [{epoch + 1}/{epochs}], Avg Train Loss: {train_epoch_loss:.4f}, Avg Test Loss: {val_epoch_loss:.4f}")
        print("---\n")

    end = time.time()

    save_loss_plot(train_loss, test_loss, 'DeepLabV3_loss.png')
    save_acc_plot(iou_acc, dice_acc, 'DeepLabV3_accuracy.png')

    excel = np.array([train_loss, test_loss, iou_acc, dice_acc])
    np.savetxt('/home/usyd-04a/checkpoints/deeplab/deeplab.csv', excel, delimiter=',')

    print(f"\nTraining took: {end - start:.2f}s")

    return model


def train_one_epoch(model, criterion, optimizer, dataloader, device, print_every):
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

        # if (batch + 1) % print_every == 0:
        #     print(f"Step [{batch + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)


def test(model, criterion, dataloader, device, num_classes):
    print("[VALIDATING]")
    ious, dice_scores = list(), list()
    model.eval()
    start = time.time()
    running_loss = 0
    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            loss = criterion(prediction, labels)
            running_loss += loss.item()
            prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)  # (batch_size, w, h)
            labels = labels.argmax(dim=1)  # (batch_size, w, h)
            iou = jaccard_index(prediction, labels, num_classes=num_classes).item()
            dice_score = dice(prediction, labels, num_classes=num_classes, ignore_index=0).item()
            ious.append(iou), dice_scores.append(dice_score)

    end = time.time()

    test_loss = running_loss / len(dataloader)
    iou_acc = np.mean(ious)
    dice_acc = np.mean(dice_scores)

    print(f"Accuracy: mIoU= {iou_acc * 100:.3f}%, dice= {dice_acc * 100:.3f}%")

    return test_loss, iou_acc, dice_acc


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="path to directory containing test and train images")
    parser.add_argument("-c", "--checkpoint",
                        help="filename for model checkpoint to be saved as")
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

def main():
    args = command_line_args()

    wandb.init(project="DeepLabV3", entity="usyd-04a", config=wandb.config, dir="./wandb_data")

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

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

    train_img_dir = os.path.join(args.data_dir, 'train/images_tiled')
    train_mask_dir = os.path.join(args.data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test/masks_tiled')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train_dataset = PlanesDataset(img_dir=train_img_dir, mask_dir=train_mask_dir, transform=transform)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], num_workers=2)

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', num_classes=HYPER_PARAMS['NUM_CLASSES']), device_ids=device_ids).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['LR'])
    loss_fn = nn.BCEWithLogitsLoss()



    train(model=model,
          criterion=loss_fn,
          optimizer=optimizer,
          train_loader=train_loader,
          test_loader=test_loader,
          device=device,
          epochs=HYPER_PARAMS['EPOCHS'],
          print_every=10,
          num_classes=HYPER_PARAMS['NUM_CLASSES'])

    save_model_2(model=model,
                 epochs=HYPER_PARAMS['EPOCHS'],
                 optimizer=optimizer,
                 criterion=loss_fn,
                 batch_size=HYPER_PARAMS['BATCH_SIZE'],
                 lr=HYPER_PARAMS['LR'],
                 filename=args.checkpoint)


if __name__ == "__main__":
    main()