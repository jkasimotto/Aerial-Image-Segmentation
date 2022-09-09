# COMP3888_W08_02
This project implements four image segmentation models (FCN, DeepLabV3, UNET, MaskRCNN) to be trained on the [Rareplanes dataset](https://www.cosmiqworks.org/rareplanes/).

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