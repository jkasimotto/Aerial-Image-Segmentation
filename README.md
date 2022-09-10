# Build the best AI: Aerial Imaging
USYD-04A / COMP3888_W08_02

---

## Overview
This project investigates the [Rareplanes dataset](https://www.cosmiqworks.org/rareplanes/) and benchmarks the
performance of various semantic / instance segmentation machine learning models.

### Model Catalog
- [x] Fully Convolutional Network (FCN)
- [x] DeepLabV3
- [x] UNet
- [x] Mask R-CNN

### The Rareplanes Dataset
The Rareplanes dataset is an open-source dataset from CosmiQ Works and AI.Reverie which incorporates
both real and synthetic data of satellite imagery. The satellite images are aerial images of various
aircraft, i.e. jets and passenger planes. Our project focuses on the use of the synthetically generated images.

![Image](assets/rareplanes_synthetic.png "Rareplanes synthetic data example")

---

## Getting Started

Install the repository, create a virtualenvironment and run 
```
pip install -r requirements.txt
``` 

---

## Download the Dataset
Follow the instructions in `rareplanes/README.md` to download the Rareplanes dataset. 

---

## Preprocess images
Follow the instructions in `preprocessing/README.md` to prepare the images for image segmentation.

---

## Using models 
Each model directory contains a `README.md` explaining how to perform training and inference. Graphical outputs of the models  are saved to `./outputs` and `.pth` checkpoint files are saved in `./checkpoints`.