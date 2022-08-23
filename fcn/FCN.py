import torch
from torchmetrics.functional import jaccard_index
from plot_helper import *
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader
from torch import nn
from dataset import PlanesDataset
import numpy as np


def train(model, criterion, optimizer, dataloader, device, epochs=1, print_every=10):
    avg_loss_list = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        print(f"Epochs [{epoch + 1}/{epochs}], Avg Loss: {running_loss / len(dataloader):.4f}")
        avg_loss_list.append(running_loss / len(dataloader))

    plot_loss(avg_loss_list)

    return model


def test(model, dataloader, device, num_classes):
    ious = list()
    model.eval()
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            _prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)  # (batch_size, w, h)
            labels = labels.argmax(dim=1)  # (batch_size, w, h)
            iou = jaccard_index(_prediction, labels, num_classes=num_classes).item()
            ious.append(iou)

    print(f'mIoU: {np.mean(ious) * 100:.3f}%')


def main():
    # Needed to download model from internet
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()}')

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'NUM_CLASSES': 2,
        'BATCH_SIZE': 3,
        'NUM_WORKERS': 2,
        'LR': 0.001,
        'EPOCHS': 1,
    }

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = '../../dataset/images'
    mask_dir = '../../dataset/masks'
    test_img_dir = '../../dataset/images'
    test_mask_dir = '../../dataset/masks'

    dataset = PlanesDataset(img_dir=img_dir, mask_dir=mask_dir)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir)
    train_loader = DataLoader(dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], num_workers=2)

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    model = fcn_resnet50(num_classes=HYPER_PARAMS['NUM_CLASSES']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['LR'])
    loss_fn = nn.BCEWithLogitsLoss()

    train(model=model,
          criterion=loss_fn,
          optimizer=optimizer,
          dataloader=train_loader,
          device=device,
          epochs=HYPER_PARAMS['EPOCHS'],
          print_every=10)

    test(model=model,
         dataloader=test_loader,
         device=device,
         num_classes=HYPER_PARAMS['NUM_CLASSES'])


if __name__ == "__main__":
    main()
