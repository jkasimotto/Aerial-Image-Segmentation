# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image
from dataset import PlanesDataset
import argparse

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from engine import *

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


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
                        help="number of classes for semantic segmentation")
    args = parser.parse_args()
    return args


def main():
    args = command_line_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ----------------------
    # DEFINE HYPER PARAMETERS
    # ----------------------

    HYPER_PARAMS = {
        'NUM_CLASSES': args.num_classes,
        'BATCH_SIZE': args.batch_size,
        'NUM_WORKERS': args.workers,
        'LR': args.learning_rate,
        'EPOCHS': args.epochs,
    }

    # ----------------------
    # CREATE DATASET
    # ----------------------

    img_dir = os.path.join(args.data_dir, 'train_small/images_tiled')
    mask_dir = os.path.join(args.data_dir, 'train_small/greyscale_masks_tiled')
    test_img_dir = os.path.join(args.data_dir, 'test_small/images_tiled')
    test_mask_dir = os.path.join(args.data_dir, 'test_small/greyscale_masks_tiled')

    # our dataset has two classes only - background and plane
    num_classes = HYPER_PARAMS['NUM_CLASSES']
    # use our dataset and defined transformations
    dataset = PlanesDataset(img_dir, mask_dir, get_transform(train=True))
    dataset_test = PlanesDataset(test_img_dir, test_mask_dir, get_transform(train=False))

    num_workers = HYPER_PARAMS['NUM_WORKERS']
    batch_size = HYPER_PARAMS['BATCH_SIZE']
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    lr = HYPER_PARAMS['LR']
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = HYPER_PARAMS['EPOCHS']

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    # make the plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        loss_plot, color='red', linestyle='-',
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("/loss_plot")
    print("That's it!")
    

if __name__ == "__main__":
    main()
