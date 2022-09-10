import matplotlib.pyplot as plt
import torch
import os

class SaveBestModel:

    def __init__(self, checkpoint_dir, checkpoint_name):
        self.best_valid_loss = float('inf')
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss:.3f}")
            print(f"Saving best model for epoch: {epoch + 1}")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, os.path.join(self.checkpoint_dir, self.checkpoint_name))

    def save_loss_plot(self, train_loss, test_loss, filename):
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        plt.savefig(os.path.join(self.checkpoint_dir, filename))


    def save_acc_plot(self, iou_acc, dice_acc, filename):
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        plt.savefig(os.path.join(self.checkpoint_dir, filename))


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


def save_model(model, epochs, optimizer, criterion, batch_size, lr, checkpoint_dir, filename):
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'batch_size': batch_size,
        'lr': lr,
    }, path)
    print(f"\nFinal model saved to {path}")
