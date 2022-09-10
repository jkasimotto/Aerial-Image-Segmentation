# U-Net
This model is based on the paper
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

---

## Training The Model

### Data Structure
The training and test images need to be structured in a specific way for the model run correctly.
The folder structure needs to be as follows:
```text
path/to/dir
    |--- train
        |--- images_tiled  # contains training images
        |--- greyscale_masks_tiled   # contains training labels
    |--- test
        |--- images_tiled  # contains test images
        |--- greyscale_masks_tiled   # contains test labels
```
**NOTE**: directory names must be the same as outlined above

### Usage
```commandline
usage: train.py [-h] [-b BATCH_SIZE] [-lr LEARNING_RATE] [-e EPOCHS] [-w WORKERS] [-n NUM_CLASSES] data_dir checkpoint

positional arguments:
  data_dir              path to directory containing test and train images
  checkpoint            path to directory for model checkpoint to be saved

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        dataloader batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate to be applied to the model
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train the model for
  -w WORKERS, --workers WORKERS
                        number of workers used in the dataloader
  -n NUM_CLASSES, --num-classes NUM_CLASSES
                        number of classes for semantic segmentation

```
### Outputs
Model checkpoints and graphs will be saved in the `checkpoint` directory. Five outputs should be produced which include
the following:

* 'unet_loss.pth' - model with the best validation from all epochs
* 'unet_acc.pth' - model with the best accuracy from all epochs
* 'unet_final_epoch.pth' - model after all epochs
* 'unet_loss.png' - graph of training loss and validation
* 'unet_accuracy' - graph of mIoU and Dice

---

## Making Predictions (Inference)
After training the model on a dataset, predictions can bew made on a set of images using the `inference.py` script.
This will display the original image with the plane mask overlay.

### Usage
```commandline
usage: inference.py [-h] [-i INDEX] model image_dir

positional arguments:
  model                 checkpoint file for pretrained model
  image_dir             path to directory containing images to run through the model

optional arguments:
  -h, --help            show this help message and exit
  -i INDEX, --index INDEX

```
Example
```commandline
python inference.py ./checkpoints/unet.pth /home/usyd-04a/synthetic/test/images/
```