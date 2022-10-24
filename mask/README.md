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
  project-name: Mask-Benchmark
```

The various speed up techniques such as AMP, channels last memory format and distributed data parallel can be turned on in the config file. Simply set
the associated `enabled` field to `True`.
