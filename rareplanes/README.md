# Downloading Rareplanes Synthetic Data
This script downloads the synthetic data from the Rareplanes AWS s3 bucket. By default, only 10 files are downloaded.

## Usage
```commandline
usage: rareplanes_synthetic.py [-h] [-o OUTPUT_PATH] [-s DATA_SPLIT] [-n LIMIT] [-a]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        location to where synthetic data will be stored
                        DEFAULT=home directory
  -s DATA_SPLIT, --data-split DATA_SPLIT
                        ratio of training data to test data to download
                        DEFAULT=0.8
  -n LIMIT, --limit LIMIT
                        number of images to download
                        DEFAULT=10
  -a, --all             download all the files
```

## Output
A new `synthetic` directory will be created with the following structure:
```text
synthetic
    |--- train
        |--- images
        |--- masks
        |--- xmls
    |--- test
        |--- images
        |--- masks
        |--- xmls
```
The train and test directories are now ready to be preprocessed.