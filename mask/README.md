## MaskRCNN model

Training the model

Data Structure
The training and test images need to be structured in a specific way for the model to run correctly. The folder structure needs to be as follows:

```
path/to/dir
    |--- train
        |--- images_tiled  # contains training images
        |--- masks_tiled   # contains training labels
    |--- test
        |--- images_tiled  # contains test images
        |--- masks_tiled   # contains test labels
```

**NOTE: directory names must be the same as outlined above

### Usage
```commandline
usage: mask-rcnn.py [-h] config_file

positional arguments:
  config_file  path to config file

options:
  -h, --help   show this help message and exit
```

Example
```commandline
python mask-rcnn.py test_config.yaml
```

### Config File
An example config file is provided below:

```yaml
config:
  run: demo
  data-dir: /home/usyd-04a/synthetic/
  checkpoint-dir: /home/usyd-04a/checkpoints/mask
  classes: 2

hyper-params:
  batch-size: 8
  learning-rate: 0.0001
  epochs: 3
  workers: 4

amp:
  enabled: False

channels-last:
  enabled: False

distributed:
  enabled: False
  nodes: 1
  ip-address: localhost
  ngpus: null
  local-ranks: 0

cuda-graphs:
  enabled: False
  warmup-iters: 5

wandb:
  enabled: False
  project-name: Mask-Demo
```

The various speed up techniques such as AMP, channels last memory format and distributed data parallel can be turned on in the config file. Simply set
the associated `enabled` field to `True`.

### Outputs
Model checkpoints and graphs will be saved in the `checkpoint_dir` directory. Given a run name of 'mask', four outputs should be produced which include
the following:

* 'mask_acc.pth' - model with the best accuracy from all epochs
* 'mask_final_epoch.pth' - model after all epochs
* 'mask_loss.png' - graph of training loss
* 'mask_accuracy' - graph of mIoU

---

## Benchmarks
These benchmarks were performed using a subset of the Rareplanes dataset, approximately 5% of the total images.

| AMP | Channels Last | DDP | Run 1 | Run 2 | Run 3 | Average |
|:---:|:-------------:|:---:|-------|-------|-------|---------|
| [X] |      [X]      | [X] | 563   | 563   | 563   | 563     |
| [X] |               | [X] | 573   | 572   | 571   | 572     |
|     |      [X]      | [X] | 757   | 757   | 759   | 758     |
|     |               | [X] | 722   | 724   | 724   | 723     |


![Image](../assets/mask_benchmark.png "Mask R-CNN benchmark graph")

---

## Making Predictions (Inference)
After training the model on a dataset, predictions can be made on a set of images using the `inference.py` script.
This will produce a new image which is the original image with a plane mask overlay.

### Usage
```commandline
usage: inference.py [-h] [-i INDEX] model image_dir prediction_dir

positional arguments:
  model                 checkpoint file for pretrained model
  image_dir             path to directory containing images to run through the model
  prediction_dir        path to directory to save predictions made by the model

options:
  -h, --help            show this help message and exit
  -i INDEX, --index INDEX
```
Example
```commandline
python inference.py ./checkpoints/mask.pt /home/usyd-04a/synthetic/test/images/ ./predictions/
```
```commandline
python inference.py ./checkpoints/mask.pt /home/usyd-04a/synthetic/test/images/ ./predictions/ -i 1
```

### Output
![Image](../assets/mask_inference.png "FCN Prediction")
