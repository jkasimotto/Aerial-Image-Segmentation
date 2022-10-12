import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision


class PlanesDataset(Dataset):

    def __init__(self, img_dir, mask_dir, num_classes, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.num_classes = num_classes
        self.images = [f for f in os.listdir(img_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask, image = np.array(mask), np.array(image, dtype=np.float32)
        mask[mask > 0] = 1  # convert any coloured pixels to 1

        # Transform
        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            image = torchvision.transforms.ToTensor()(image)

        # Split mask into binary masks for each class
        mask = np.array(mask)
        color_ids = np.unique(mask)  # find all unique colors in mask
        # masks = (mask == color_ids[:, None, None])

        # # Add empty masks if needed for remaining classes if removed by transforms
        # if masks.shape[0] < self.num_classes:
        #     for i in range(self.num_classes - 1):
        #         x = np.expand_dims(np.zeros_like(mask), axis=0)
        #         masks = np.concatenate((masks, x))

        # masks = torch.as_tensor(masks, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)


        return image, mask


# class PlanesDataset(Dataset):

#     def __init__(self, img_dir, mask_dir, transform=None):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = [f for f in os.listdir(img_dir) if not f.startswith('.')]
#         self.as_tensor = transforms.ToTensor()

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_dir, self.images[index])
#         mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_mask.png'))

#         image = np.array(Image.open(img_path).convert("RGB")) # Used np.array to use the albumentations library.
#         mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

#         mask[mask > 0] = 1  # convert all non black colours to the 'plane' class pixel
#         # color_ids = np.unique(mask)  # find all unique colors in mask
#         # masks = mask == color_ids[:, None, None]
#         # masks = torch.as_tensor(masks, dtype=torch.float32)

#         # Transform
#         if self.transform is not None:
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]

#         return image, mask
