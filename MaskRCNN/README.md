## MaskRCNN model

Training the model

Data Structure
The training and test images need to be structured in a specific way for the model to run correctly. The folder structure needs to be as follows:

path/to/dir
    |--- train
        |--- images_tiled  # contains training images
        |--- masks_tiled   # contains training labels
    |--- test
        |--- images_tiled  # contains test images
        |--- masks_tiled   # contains test labels
        
NOTE: directory names must be the same as outlined above

Usage

usage: fcn.py [-h] [-n NUM_CLASSES] [-b BATCH_SIZE] [-w WORKERS] [-lr LEARNING_RATE] [-e EPOCHS] 

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