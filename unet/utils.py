import os

import matplotlib as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.functional import dice, jaccard_index
from tqdm import tqdm

from dataset import PlanesDataset


class SaveBestModel:

    def __init__(self):
        self.best_valid_loss = float('inf')

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss:.3f}")
            print(f"Saving best model for epoch: {epoch + 1}")
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, './checkpoints/unet.pth')


def save_loss_plot(train_loss, test_loss, filename):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        test_loss, color='red', linestyle='-',
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig(os.path.join('./outputs', filename))


def save_acc_plot(iou_acc, dice_acc, filename):
    plt.figure(figsize=(10, 7))
    plt.plot(
        iou_acc, color='orange', linestyle='-',
        label='mIoU'
    )
    plt.plot(
        dice_acc, color='red', linestyle='-',
        label='dice score'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig(os.path.join('./outputs', filename))


def plot_pred(prediction):
    prediction = torch.sigmoid(prediction) > 0.5
    for x in prediction:
        plt.imshow(x.cpu())
        plt.show()


def plot_loss(losses):
    x = list(range(1, len(losses) + 1))
    plt.plot(x, losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.show()


def save_model_2(model, epochs, optimizer, criterion, batch_size, lr, filename):
    path = os.path.join('./checkpoints', filename)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'batch_size': batch_size,
        'lr': lr,
    }, path)
    print(f"\nFinal model saved to {path}")


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    train_ds = PlanesDataset(
        img_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = PlanesDataset(
        img_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader