from torchmetrics.functional import jaccard_index, dice
from utils import *
import torchvision
from torchvision.models.segmentation import fcn_resnet101
from torch.utils.data import DataLoader
import torch.profiler
from torch import nn
from dataset import PlanesDataset
import numpy as np
from tqdm import tqdm
import time
import argparse
import cv2
from glob import glob
from torchvision.ops import masks_to_boxes
from google.colab.patches import cv2_imshow


def main():
   

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU avaliable: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

                # ----------------------
                    # DEFINE HYPER PARAMETERS
                        # ----------------------

    HYPER_PARAMS = {
    'NUM_CLASSES': 2,
    'BATCH_SIZE': 5,
    'NUM_WORKERS': 1,
    'LR': 0.01,
    'EPOCHS': 5,}


    listTemp = [2, 3, 5, 6, 7]
    image_size = [512, 512]

    # CREATING DATASET

    print(glob('sample_data/images/*'))
    batch_Imgs,batch_Data=[],[]
    for image in glob('sample_data/images/*'):
    name = image.split('/')[2].split('.')[0]
    im = cv2.imread(image)
    mask_name = 'sample_data/masks/'+ name+'_mask'+'.png'
    mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    data = {}
    boxes = torch.zeros([len(bounding_boxes),4],dtype=torch.float32)
    for i,bound in enumerate(bounding_boxes):
        x,y,w,h = bound
        boxes[i] = torch.tensor([x, y, x+w, y+h]) # remember u can change these might fix future issues

    data["boxes"] =  boxes
    data["labels"] =  torch.ones((len(bounding_boxes),), dtype=torch.int64)   
    data["masks"] = torch.as_tensor([mask],dtype=torch.uint8) 
    img = torch.as_tensor(im, dtype=torch.float32)
    batch_Imgs.append(img)
    batch_Data.append(data)
    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)  


    # MODEL

    model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)
    model.to(device)  
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    model.train()

    # TRAINING

    for i in range(100):
        images = [image.to(device) for image in batch_Imgs]
        targets = [{k:v.to(device) for k,v in t.items()} for t in batch_Data]
        # for t in batch_Data:
        #   for key,value in t.items():
        #     print(key)
        #     print(value)
        #     targets.append({key:value.to(device)})
        optimizer.zero_grad()
        loss_dict = model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(i,'loss:', losses.item())
        # if i%500==0:
        #   torch.save(model.state_dict(), str(i)+".torch")
    torch.save(model.state_dict(), "save.torch")

    # TEST ON ONE IMAGE

    model.eval()
    import random
    # images=cv2.imread("/Users/aashika/Desktop/COMP3888Sep/comp3888_w08_02/dataset/test/images/Atlanta_Airport_0_0_10_918.png")
    images = cv2.imread("sample_data/images/Atlanta_Airport_0_0_100_2_tile_2.png")
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)
    with torch.no_grad():
            pred = model(images)
    im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
    im2 = im.copy()
    scr = None
    for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][i,0].detach().cpu().numpy()
            scr=pred[0]['scores'][i].detach().cpu().numpy()
            if scr>0.8 :
                im2[:,:,0][msk>0.5] = random.randint(0,255)
                im2[:, :, 1][msk > 0.5] = random.randint(0,255)
                im2[:, :, 2][msk > 0.5] = random.randint(0, 255)
    
    cv2.imshow(str(scr), np.hstack([im,im2]))
    cv2.waitKey()
                                                                

if __name__ == "__main__":
    main()
