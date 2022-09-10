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
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_greyscale_mask.png'))

        image = np.array(Image.open(img_path).convert("RGB")) # Used np.array to use the albumentations library.
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        mask[mask > 0] = 1  # convert all non black colours to the 'plane' class pixel
        # color_ids = np.unique(mask)  # find all unique colors in mask
        # masks = mask == color_ids[:, None, None]
        # masks = torch.as_tensor(masks, dtype=torch.float32)

        # Transform
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
