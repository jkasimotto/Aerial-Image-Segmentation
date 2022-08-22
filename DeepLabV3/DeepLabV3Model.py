import ssl
import torch
import matplotlib.pyplot as plt
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

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------
    
    HYPER_PARAMS = {
        'NUM_CLASSES': 2,
        'BATCH_SIZE': 5,
        'NUM_WORKERS': 2,
        'LR': 0.001,
        'EPOCHS': 30,
    }

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = 'images_tiled'
    mask_dir = 'masks_tiled'
    dataset = PlanesDataset(img_dir=img_dir, mask_dir=mask_dir)
    train_loader = DataLoader(dataset, batch_size=HYPER_PARAMS['BATCH_SIZE'], shuffle=True, num_workers=2)

    # ----------------------
    # DEFINE MODEL
    # ----------------------

    #model = fcn_resnet50(num_classes=HYPER_PARAMS['NUM_CLASSES']).to(device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', num_classes=HYPER_PARAMS['NUM_CLASSES']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMS['LR'])
    loss_fn = nn.BCEWithLogitsLoss()

    # ----------------------
    # TRAINING LOOP
    # ----------------------

    model.train()
    print_every = 5
    # Data to graph average loss
    avgLossList = []
    for epoch in range(HYPER_PARAMS['EPOCHS']):
        running_loss = 0
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)['out']

            # print first and second layer
            # plt.imshow(prediction[0][0].cpu().detach().numpy())
            # plt.show()
            # plt.imshow(prediction[0][1].cpu().detach().numpy())
            # plt.show()

            loss = loss_fn(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            if (batch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{HYPER_PARAMS['EPOCHS']}], Step [{batch + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"Epochs [{epoch + 1}/{HYPER_PARAMS['EPOCHS']}], Avg Loss: {running_loss / len(train_loader):.4f}")
        avgLossList.append(running_loss / len(train_loader))
    
    for batch, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      prediction = model(images)['out']
      for single_prediction in prediction:
        # print first and second layer
        plt.imshow(single_prediction[0].cpu().detach().numpy())
        plt.show()
        plt.imshow(single_prediction[1].cpu().detach().numpy())
        plt.show()


    plt.plot(list(range(1, len(avgLossList) + 1)), avgLossList)
    plt.show()