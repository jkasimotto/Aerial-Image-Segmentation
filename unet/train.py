import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
    test
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False

# 1.  Directories are different
# 2. Can train on C

DIR = "/home/julianotto/workspace/comp3888_w08_02/"
TRAIN_IMG_DIR = DIR + "local/data/synthetic/train/images_tiled"
TRAIN_MASK_DIR = DIR + "local/data/synthetic/train/masks_tiled"
VAL_IMG_DIR = DIR + "local/data/synthetic/val/images_tiled"
VAL_MASK_DIR = DIR + "local/data/synthetic/val/masks_tiled"


def train(model, criterion, optimizer, dataloader, scaler, epochs=1, print_every=10):
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    avg_loss_list = []
    model.train()
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0
        for batch, (images, labels) in enumerate(dataloader):
            images = images.to(device=DEVICE)
            labels = labels.float().unsqueeze(1).to(device=DEVICE) # Add a channel dimension
            # forward
            with torch.cuda.amp.autocast(): # Allows mixed precision operations for forward pass.
                predictions = model(images)
                loss = criterion(predictions, labels)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward() # Updates mixed precision weights
            scaler.step(optimizer) # 
            scaler.update()
            running_loss += loss.item()
            # Print batch messages
            if (batch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        # Print epoch messages
        print(f"Epochs [{epoch + 1}/{epochs}], Avg Loss: {running_loss / len(dataloader):.4f}")
        avg_loss_list.append(running_loss / len(dataloader))
        # Save checkpoint each epoch.
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
    # Finish training
    end = time.time()
    print(f"\nTraining took: {end - start:.2f}s")
    return model

def main():
    train_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0,0.0,0.0],
            std=[1.0,1.0,1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0,0.0,0.0],
            std=[1.0,1.0,1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # If the forward pass of an operation has float16 inputs, small gradients may not be 
    # representable in float16 and will 'underflow' to 0. Gradient Scaling prevents this.
    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    scaler = torch.cuda.amp.GradScaler()
    train(model, criterion, optimizer, train_loader, scaler, NUM_EPOCHS)
    save_predictions_as_imgs(
        val_loader, model, folder="saved_images/", device=DEVICE
    )

# Do this so on Windows there are no issues when using NUM_WORKERS
if __name__ == "__main__":
    main()