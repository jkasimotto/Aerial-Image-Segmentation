import argparse
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
from dataset import PlanesDataset
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet101
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
import torch.distributed as dist
import numpy
import random

def augmentations():
    train_transforms = A.Compose([
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()])

    test_transforms = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()])

    return train_transforms, test_transforms


def my_collate_fn(batch):
    images, labels = [], []
    for img, mask in batch:
        images.append(img)
        labels.append(mask)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels