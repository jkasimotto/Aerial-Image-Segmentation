import ssl
import torch
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader
from torch import nn
from dataset import PlanesDataset

if __name__ == "__main__":
    # Needed to download model from internet
    ssl._create_default_https_context = ssl._create_unverified_context

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()}')

    # Define hyper parameters
    HYPER_PARAMS = {
        'NUM_CLASSES': 2,
        'BATCH_SIZE': 2,
        'NUM_WORKERS': 2,
        'LR': 0.001,
        'EPOCHS': 1,
    }

    # Create the dataset
    img_dir = 'images_tiled'
    mask_dir = 'masks_tiled'
    dataset = PlanesDataset(img_dir=img_dir, mask_dir=mask_dir)
    train_loader = DataLoader(dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], shuffle=True, num_workers=2)

    # Define model
    model = fcn_resnet50(num_classes=HYPER_PARAMS['NUM_CLASSES']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['LR'])
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    print_every = 5
    for epoch in range(HYPER_PARAMS['EPOCHS']):
        running_loss = 0
        for batch, (images, labels) in enumerate(train_loader):
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
