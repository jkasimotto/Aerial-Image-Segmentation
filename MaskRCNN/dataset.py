import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlanesDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))
        # converts a PIL Image to a torch.FloatTensor
        # of shape [C, H, W] in the range [0.0, 1.0]
        self.as_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        # load image and mask paths
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        # open the original image and conver it to RGB
        img = self.as_tensor(Image.open(img_path).convert("RGB"))
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("L")
        # convert the PIL Image into a numpy array
        mask  = seg_mask = np.array(mask)
        seg_mask[seg_mask > 0] = 1  # convert white pixels to 1
        # instances are encoded as different colors
        obj_ids = np.unique(mask) # unique elements of mask in ascending order
        seg_obj_ids = np.unique(mask)  # unique elements of mask in ascending order
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        seg_obj_ids = seg_obj_ids[1:]


        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        seg_masks = seg_mask == seg_obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        boxes = []
        for mask in masks:
            pos = np.asarray(mask).nonzero() # returns an array of the co-ordinates of pixels that are non-zero
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(obj_ids), dtype=torch.int64) # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        seg_masks = torch.as_tensor(seg_masks, dtype=torch.uint8)

        # create the target dict for training
        target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "seg_mask": seg_masks,
        }

        return img, target

    def __len__(self):
        return len(self.imgs)
