import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import PlanesDataset
from tqdm import tqdm
from torchmetrics.functional import jaccard_index, dice


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("==> Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint, model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

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

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
    
    model.train()

def test(model, dataloader, device, num_classes):
    print("\n=================")
    print("| Testing Model |")
    print("=================\n")

    ious, dice_scores = list(), list()
    num_correct = 0
    num_pixels = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            prediction = torch.sigmoid(model(images))
            prediction = (prediction > 0.5).float()
            num_correct += (prediction == labels).sum()
            num_pixels += torch.numel(prediction)

            # iou = jaccard_index(prediction, labels, num_classes=num_classes).item()
            # dice_score = dice(prediction, labels, num_classes=num_classes, ignore_index=0).item()
            # ious.append(iou), dice_scores.append(dice_score)

    print("\n=================")
    print("| Model Results |")
    print("-----------------")
    print(f'| acc: {num_correct / num_pixels}  |')
    # print(f'| mIoU: {np.mean(ious) * 100:.3f}%  |')
    # print(f'| dice: {np.mean(dice_scores) * 100:.3f}%  |')
    print("=================\n")