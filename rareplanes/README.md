# Downloading Rareplanes Synthetic Data
This script downloads the synthetic data from the Rareplanes AWS s3 bucket. By default, only 10 files are downloaded.

## Usage
```commandline
python3 rareplanes_synthetic.py
```

### Optional Arg: Output Path
The location where to download the files can be specified using the `-o` or `--output-path`
arguments. By default, the files will be downloaded in the root directory in a folder called 'synthetic'.

### Optional Arg: File Limit
The number of files downloaded can be limited by using the `-n` or `--limit` arguments. By default, 10 files will
be downloaded.

### Flag: Download All
To download all the files approximately 211GB add the `-a` flag. This will override the limit flag.
