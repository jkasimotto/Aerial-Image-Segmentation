import matplotlib.pyplot as plt
import torch


def plot_pred(prediction):
    prediction = prediction.softmax(dim=1).argmax(dim=1) > 0
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


def save_model(model, path):
    torch.save(model, path)
    print(f"\nModel saved to {path}")