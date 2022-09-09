import matplotlib.pyplot as plt
import torch
import os


class ModelAnalyzer:

    def __init__(self, checkpoint_dir):
        self.best_valid_loss = float('inf')
        self.best_accuracy = 0
        self.checkpoint_dir = checkpoint_dir

    def save_best_model(self, current_valid_loss, current_accuracy, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss:.3f}")
            print(f"Saving best loss model for epoch: {epoch + 1}")
            torch.save({
                'epoch': epoch + 1,
                'accuracy': self.best_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, os.path.join(self.checkpoint_dir, 'fcn_loss.pth'))
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            print(f"Best accuracy (mIoU): {self.best_accuracy:.3f}")
            print(f"Saving best accuracy model for epoch: {epoch + 1}")
            torch.save({
                'epoch': epoch + 1,
                'accuracy': self.best_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, os.path.join(self.checkpoint_dir, 'fcn_acc.pth'))

    def save_loss_plot(self, train_loss, test_loss):
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
        plt.savefig(os.path.join(self.checkpoint_dir, 'fcn_loss.png'))

    def save_acc_plot(self, iou_acc, dice_acc):
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
        plt.savefig(os.path.join(self.checkpoint_dir, 'fcn_acc.png'))

    def save_model(self, model, epochs, optimizer, criterion, batch_size, lr):
        path = os.path.join(self.checkpoint_dir, 'fcn_final_epoch.pth')
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'batch_size': batch_size,
            'lr': lr,
        }, path)
        print(f"\nFinal model saved to {path}")
