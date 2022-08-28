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

def train(model, criterion, optimizer, dataloader, scaler, device, epochs=1, print_every=10):
    print("\n==================")
    print("| Training Model |")
    print("==================\n")

    avg_loss_list = []
    model.train()
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0
        for batch, (images, labels) in enumerate(dataloader):
            images = images.to(device=device)
            labels = labels.float().unsqueeze(1).to(device=device) # Add a channel dimension
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
        print(
            f"Epochs [{epoch + 1}/{epochs}], Avg Loss: {running_loss / len(dataloader):.4f}")
        avg_loss_list.append(running_loss / len(dataloader))
        # Save checkpoint each epoch.
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f"./checkpoints/epoch-{epoch}-checkpoint.pth")
    # Finish training
    end = time.time()
    print(f"\nTraining took: {end - start:.2f}s")
    return model


def main():

    LOAD_MODEL = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'NUM_CLASSES': 2,
        'BATCH_SIZE': 16,
        'NUM_WORKERS': 2,
        'LR': 0.001,
        'EPOCHS': 15,
        'PIN_MEMORY': True
    }

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = '/home/usyd-04a/synthetic/train/images_tiled'
    mask_dir = '/home/usyd-04a/synthetic/train/masks_tiled/'
    test_img_dir = '/home/usyd-04a/synthetic/test/images_tiled/'
    test_mask_dir = '/home/usyd-04a/synthetic/test/masks_tiled/'

    # Augmentations to training set
    train_transforms = A.Compose([
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    # Augmentations to test set
    test_transforms = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    train_loader, val_loader = get_loaders(
        img_dir,
        mask_dir,
        test_img_dir,
        test_mask_dir,
        HYPER_PARAMS['BATCH_SIZE'],
        train_transforms,
        test_transforms,
        HYPER_PARAMS['NUM_WORKERS'],
        HYPER_PARAMS['PIN_MEMORY']
    )

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    model = UNET(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=HYPER_PARAMS["LR"])
    # If the forward pass of an operation has float16 inputs, small gradients may not be 
    # representable in float16 and will 'underflow' to 0. Gradient Scaling prevents this.
    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    scaler = torch.cuda.amp.GradScaler()

    train(model,
          criterion=criterion,
          optimizer=optimizer,
          dataloader=train_loader,
          scaler=scaler,
          device=device,
          epochs=HYPER_PARAMS["EPOCHS"],
          print_every=30)

    save_predictions_as_imgs(
        val_loader, model, folder="./saved_images/", device=device
    )


# Do this so on Windows there are no issues when using NUM_WORKERS
if __name__ == "__main__":
    main()
