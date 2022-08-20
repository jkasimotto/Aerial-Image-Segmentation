import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class PlanesDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        self.as_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask)
        mask[mask > 0] = 1  # convert and non-black pixels to 1
        color_ids = np.unique(mask)  # find all unique colors in mask
        masks = mask == color_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.float32)

        return self.as_tensor(image), masks
