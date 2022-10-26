# Fully Convolutional Network (FCN)
This model is based on the paper
[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).

---

## Training The Model

### Data Structure
The training and test images need to be structured in a specific way for the model run correctly.
The folder structure needs to be as follows:
```text
path/to/dir
    |--- train
        |--- images_tiled  # contains training images
        |--- masks_tiled   # contains training labels
    |--- test
        |--- images_tiled  # contains test images
        |--- masks_tiled   # contains test labels
```
**NOTE**: directory names must be the same as outlined above

### Usage
```commandline
usage: fcn.py [-h] config_file

positional arguments:
  config_file  path to config file

options:
  -h, --help   show this help message and exit
```

Example
```commandline
python fcn.py test_config.yaml
```

### Config File
An example config file is provided below:

```yaml
config:
  run: demo
  data-dir: /home/usyd-04a/synthetic/
  checkpoint-dir: /home/usyd-04a/checkpoints/fcn
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
  project-name: FCN-Demo
```

The various speed up techniques such as AMP, channels last memory format and distributed data parallel can be turned on in the config file. Simply set
the associated `enabled` field to `True`.

### Outputs
Model checkpoints and graphs will be saved in the `checkpoint_dir` directory. Given a run name of 'fcn', five outputs should be produced which include
the following:

* 'fcn_loss.pth' - model with the best validation from all epochs
* 'fcn_acc.pth' - model with the best accuracy from all epochs
* 'fcn_final_epoch.pth' - model after all epochs
* 'fcn_loss.png' - graph of training loss and validation
* 'fcn_accuracy' - graph of mIoU and Dice

---

## Benchmarks
These benchmarks were performed using a subset of the Rareplanes dataset, approximately 5% of the total images.

| AMP | Channels Last | DDP | Run 1 | Run 2 | Run 3 | Average |
|:---:|:-------------:|:---:|-------|-------|-------|---------|
| [X] |      [X]      | [X] | 377   | 379   | 380   | 379     |
| [X] |      [X]      |     | 416   | 416   | 419   | 417     |
| [X] |               | [X] | 366   | 368   | 369   | 368     |
| [X] |               |     | 446   | 445   | 444   | 445     |
|     |      [X]      | [X] | 560   | 560   | 561   | 560     |
|     |      [X]      |     | 633   | 635   | 634   | 634     |
|     |               | [X] | 531   | 531   | 531   | 531     |
|     |               |     | 637   | 637   | 637   | 637     |


![Image](../assets/fcn_benchmark.png "FCN benchmark graph")

### CUDA Graphs Benchmarks
Our CUDA graphs implementation does not have any validation per epoch thus separate
benchmarking runs were recorded. These times exclude the initial warm up iterations
required by CUDA graphs.

| CUDA Graphs | Channels Last | DDP | Run 1 | Run 2 | Run 3 | Average |
|:-----------:|:-------------:|:---:|-------|-------|-------|---------|
|     [X]     |               | [X] | 412   | 415   | 415   | 414     |
|     [X]     |      [X]      | [X] | 464   | 463   | 463   | 463     |
|             |               | [X] | 428   | 431   | 432   | 430     |
|             |      [X]      | [X] | 484   | 484   | 484   | 484     |

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
python inference.py ./checkpoints/fcn.pt /home/usyd-04a/synthetic/test/images/ ./predictions/
```
```commandline
python inference.py ./checkpoints/fcn.pt /home/usyd-04a/synthetic/test/images/ ./predictions/ -i 1
```

### Output
![Image](../assets/fcn_inference.png "FCN Prediction")