import os
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from keras.datasets import mnist
from torchvision import transforms
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

img_dir = '/Users/aashika/Desktop/COMP3888Clone/comp3888_w08_02/train/images_tiled/'
x_train = []

# Atlanta_Airport_0_0_100_1836_tile_1.
as_tensor = transforms.ToTensor()
for i in range(1, 6):
    name = "Atlanta_Airport_0_0_100_1836_tile_" + str(i) + ".png"
    img_path = os.path.join(img_dir, name)
    # print(img_path)
    image = Image.open(img_path).convert("RGB")
    image = as_tensor(image)
    # print(image)
    x_train.append(image)


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)

# img_path = os.path.join(self.img_dir, self.images[index])
#         mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_mask.png'))

#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         mask = np.array(mask)
#         mask[mask == 255] = 1  # convert white pixels to 1
#         color_ids = np.unique(mask)  # find all unique colors in mask
#         masks = mask == color_ids[:, None, None]
#         masks = torch.as_tensor(masks, dtype=torch.float32)

#         # Transform
#         if self.transform:
#             image = self.transform(image)
#         else:
#             image = self.as_tensor(image)

