import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class InferenceDataset(Dataset):

    def __init__(self, img_dir, start_idx, end_idx, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        self.images = self.images[start_idx:end_idx]
        self.as_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")


        # Transform
        if self.transform:
            normalised_image = self.transform(image)
            image = self.as_tensor(image)

        else:
            normalised_image = self.as_tensor(image)
            image = self.as_tensor(image)

        return normalised_image, image