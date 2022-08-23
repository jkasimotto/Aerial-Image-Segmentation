import ssl
import torch
from plot_helper import *
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader
from torch import nn
from dataset import PlanesDataset
import numpy as np


def train(model, dataloader, device, epochs=1, print_every=10):
    avg_Loss_list = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            loss = loss_fn(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{HYPER_PARAMS['EPOCHS']}], Step [{batch + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"Epochs [{epoch + 1}/{HYPER_PARAMS['EPOCHS']}], Avg Loss: {running_loss / len(train_loader):.4f}")
        avg_Loss_list.append(running_loss / len(dataloader))

    plot_loss(avg_Loss_list)

    return model


def test(model, dataloader, device):
    ious = []
    model.eval()
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']
            _prediction = prediction.softmax(dim=1).argmax(dim=1).squeeze(1)
            labels = labels.argmax(dim=1)

            iou_list = list()
            present_iou_list = list()
            pred = _prediction.view(-1)
            label = labels.view(-1)

            for sem_class in range(2):
                pred_inds = (pred == sem_class)
                target_inds = (label == sem_class)
                if target_inds.long().sum().item() == 0:
                    iou_now = float('nan')
                else:
                    intersection_now = (pred_inds[target_inds]).long().sum().item()
                    union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                    iou_now = float(intersection_now) / float(union_now)
                    present_iou_list.append(iou_now)
                iou_list.append(iou_now)
            print(np.mean(present_iou_list))
            ious.append(np.mean(present_iou_list))

    print(f'mIoU: {np.mean(ious) * 100:.3f}%')


if __name__ == "__main__":
    # Needed to download model from internet
    ssl._create_default_https_context = ssl._create_unverified_context

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()}')

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'NUM_CLASSES': 2,
        'BATCH_SIZE': 2,
        'NUM_WORKERS': 2,
        'LR': 0.001,
        'EPOCHS': 1,
    }

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = 'images'
    mask_dir = 'masks'
    test_img_dir = 'images'
    test_mask_dir = 'masks'
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

    train(model=model, dataloader=train_loader, device=device, epochs=HYPER_PARAMS['EPOCHS'], print_every=10)
    test(model=model, dataloader=test_loader, device=device)
