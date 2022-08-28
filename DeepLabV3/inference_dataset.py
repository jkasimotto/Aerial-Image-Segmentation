import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class InferenceDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        self.as_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")


        # Transform
        if self.transform:
            image = self.transform(image)
        else:
            image = self.as_tensor(image)

        return image