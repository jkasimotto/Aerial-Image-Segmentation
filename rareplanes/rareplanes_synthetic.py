import boto3
from botocore.handlers import disable_signing
from botocore import UNSIGNED
from botocore.config import Config
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import math


def main():
    # Setup command line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path",
                        help="location to where synthetic data will be stored", default=Path.home())
    parser.add_argument("-s", "--data-split",
                        help="ratio of training data to test data to download", default=0.8, type=float)
    parser.add_argument("-n", "--limit", help="number of images to download", type=int, default=10)
    parser.add_argument("-a", "--all", help="download all the files", action='store_true')
    args = parser.parse_args()

    if args.all:
        print("To download all the data run the following command:")
        print(f"aws s3 cp --recursive --no-sign-request s3://rareplanes-public/synthetic/ {args.output_path}")
        return

    if args.data_split <= 0 or args.data_split > 1:
        print("Invalid data split argument. Valid range 0 < x <= 1")
        return

    num_train_files = math.ceil(args.limit * args.data_split)
    num_test_files = args.limit - num_train_files

    # Get filenames from s3 bucket
    bucket_name = 'rareplanes-public'
    s3 = boto3.resource('s3')
    s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    bucket = s3.Bucket(bucket_name)

    # Get test image filenames
    test_files = []
    for obj in bucket.objects.filter(Prefix="synthetic/test/images/").limit(num_test_files):
        test_files.append(obj.key)

    # Get train image filenames
    train_files = []
    for obj in bucket.objects.filter(Prefix="synthetic/train/images/").limit(num_train_files):
        train_files.append(obj.key)

    all_files = [test_files, train_files]
    all_files_names = ['test', 'train']

    # Download the files

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for file_list, name in zip(all_files, all_files_names):
        print(f"Downloading {name} files")
        for image_file in tqdm(file_list):
            mask_file = image_file.replace(".png", "_mask.png").replace('/images/', '/masks/')
            xml_file = image_file.replace(".png", ".xml").replace('/images/', '/xmls/')

            files = [image_file, mask_file, xml_file]
            for f in files:
                directory_path = os.path.join(args.output_path, f.rsplit('/', 1)[0])
                os.makedirs(directory_path, exist_ok=True)

            s3.download_file(bucket_name, image_file, os.path.join(args.output_path, image_file))
            s3.download_file(bucket_name, mask_file, os.path.join(args.output_path, mask_file))
            s3.download_file(bucket_name, xml_file, os.path.join(args.output_path, xml_file))


if __name__ == "__main__":
    main()
