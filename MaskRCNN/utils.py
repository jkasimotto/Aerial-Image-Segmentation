import argparse
import matplotlib.pyplot as plt
import torch
import os
import torch.distributed as dist


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="path to directory containing test and train images")
    parser.add_argument("checkpoint",
                        help="path to directory for model checkpoint to be saved")
    parser.add_argument("-b", '--batch-size', default=16, type=int,
                        help="dataloader batch size")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,
                        help="learning rate to be applied to the model")
    parser.add_argument("-e", "--epochs", default=1, type=int,
                        help="number of epochs to train the model for")
    parser.add_argument("-w", "--workers", default=2, type=int,
                        help="number of workers used in the dataloader")
    parser.add_argument("-n", "--num-classes", default=2, type=int,
                        help="number of classes for instance segmentation")
    args = parser.parse_args()
    return args


class SaveBestModel:

    def __init__(self, checkpoint_dir):
        self.best_accuracy = 0
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, current_accuracy, epoch, model, optimizer):
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            print(f"Best accuracy (mIoU): {self.best_accuracy:.3f}")
            print(f"Saving best accuracy model for epoch: {epoch + 1}")
            torch.save({
                'epoch': epoch + 1,
                'accuracy': self.best_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.checkpoint_dir, 'mask_acc.pth'))


def save_loss_plot(train_loss, filepath):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filepath)


def save_acc_plot(iou_acc, dice_acc, filepath):
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
    plt.savefig(filepath)


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


def save_model(model, epochs, optimizer, criterion, batch_size, lr, filepath):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'batch_size': batch_size,
        'lr': lr,
    }, filepath)
    print(f"\nFinal model saved to {filepath}")


def collate_fn(batch):
    return tuple(zip(*batch))
