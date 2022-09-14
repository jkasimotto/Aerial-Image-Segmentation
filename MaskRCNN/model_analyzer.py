import matplotlib.pyplot as plt
import torch
import os


class ModelAnalyzer:

    def __init__(self, checkpoint_dir, run_name):
        self.best_accuracy = 0
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name

    def save_best_model(self, current_accuracy, epoch, model, optimizer):
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            print(f"Best accuracy (mIoU): {self.best_accuracy:.3f}")
            print(f"Saving best accuracy model for epoch: {epoch + 1}")
            torch.save({
                'epoch': epoch + 1,
                'accuracy': self.best_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.checkpoint_dir, f'self.run_name_acc.pth'))


    def save_loss_plot(self, train_loss):
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-',
            label='train loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, f'{self.run_name}_loss.png'))


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
        plt.savefig(os.path.join(self.checkpoint_dir, f'{self.run_name}_acc.png'))


    def save_model(self, model, epochs, optimizer, batch_size, lr):
        path = os.path.join(self.checkpoint_dir, f'{self.run_name}_final_epoch.pth')
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_size': batch_size,
            'lr': lr,
        }, path)
        print(f"\nFinal model saved to {path}")
