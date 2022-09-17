import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision


class PlanesDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(img_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask)
        mask[mask > 0] = 1  # convert any coloured pixels to 1
        color_ids = np.unique(mask)  # find all unique colors in mask
        masks = mask == color_ids[:, None, None]

        # Transform
        if self.transforms is not None:
            image, masks = np.array(image), np.array(masks, dtype=np.float32)
            augmentations = self.transforms(image=image, mask=masks)
            image = augmentations["image"]
            masks = augmentations["mask"]
        else:
            image = torchvision.transforms.ToTensor()(image)
            masks = torch.as_tensor(masks, dtype=torch.float32)

        return image, masks
