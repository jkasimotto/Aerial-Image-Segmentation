# Inference - Making Predictions
Allows you to make predictions on images using a pretrained model.

## Arguments
* model (str) - checkpoint file for a pretrained model
* image_dir (path) - path to a directory which contains the images which will be passed through the model

## Usage
```commandline
python inference.py ../fcn/checkpoints/fcn.pt /home/usyd-04a/synthetic/test/images/
```