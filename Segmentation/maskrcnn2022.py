# 0. Import Libraries and Mount Drive
print("0. Import Libraries and Mount Drive", flush=True)
import warnings
import logging
warnings.filterwarnings("ignore")
import os
import cv2
import sys
from tqdm import tqdm
from scipy import stats
import shutil
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label, regionprops
import tensorflow as tf
import keras
tf.get_logger().setLevel(logging.ERROR)

from imgaug import augmenters as iaa
from datetime import timedelta
import datetime
import imageio
import math
import pytz
from pytz import timezone
import imagecodecs._imcd
from tifffile import imread

from root.Mask_RCNN.mrcnn.config import Config
from root.Mask_RCNN.mrcnn import utils
from root.Mask_RCNN.mrcnn import model as modellib
from root.Mask_RCNN.mrcnn import visualize
from root.Mask_RCNN.mrcnn import postprocessing
print("##########################################################", flush=True)

imagepath=sys.argv[1]
imagename=sys.argv[2]
savepath=sys.argv[3]
weightpath=sys.argv[4]
print("imagepath is ",imagepath, flush=True)
print("imagename is ",imagename, flush=True)
print("savepath is ",savepath, flush=True)

# 1. Load and Process Images
print("1. Load and Process Images", flush=True)
img = cv2.imread(imagepath)
top_size,bottom_size,left_size,right_size = 128,128,128,128
img = cv2.copyMakeBorder(img, top_size,bottom_size,left_size,right_size, cv2.BORDER_CONSTANT, value=(0,0,0))
imggrid = []
for i in range(5):
    for j in range(5):
        imggrid.append(img[i*512:(i+1)*512,j*512:(j+1)*512])
print("##########################################################", flush=True)

# 2. Configuration
print("2. Configuration", flush=True)
# Confidence Threshold for Target Detection
DETECTION_MIN_CONFIDENCE = 0.9
# Testing Non Maximum Suppression (NMS) Threshold
DETECTION_NMS_THRESHOLD = 0.1

class NucleusConfig(Config):
    NAME = "nucleus"
    IMAGES_PER_GPU = 1  #@param {type:"integer"}
    NUM_CLASSES = 2  #@param {type:"integer"}
    DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE #@param {type:"number"}
    DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD #@param {type:"number"}
    RPN_NMS_THRESHOLD = 0.5 #@param {type:"number"}
    BACKBONE = "resnet50" #@param ["resnet50", "resnet101"] {type:"string"} 
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    MP = np.mean(img,axis=(0,1))
    MEAN_PIXEL = MP
    USE_MINI_MASK = False #@param {type:"boolean"}
    height = 128 #@param {type:"integer"}
    width = 128 #@param {type:"integer"}
    MINI_MASK_SHAPE = (height, width)  # (height, width) of the mini-mask
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 256
class NucleusInferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "none"
config = NucleusConfig()
print("##########################################################", flush=True)
        
# 3. Run the Network
print("3. Run the Network", flush=True)
Weights = "Kaggle"
if Weights == "Kaggle":
    weights_path = weightpath+'/mask_rcnn_kaggle_v1.h5' 
elif Weights == "Storm_Cell":
    weights_path = weightpath+'/mask_rcnn_nucleus_cell.h5' 
print("weights_path : ", weights_path)

config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=weights_path)
if Weights == "coco":
    model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)
results=[]
for image in imggrid:
    results.append(model.detect([image], verbose=0)[0])
print("##########################################################", flush=True)

# 4. post processing
print("4. post processing", flush=True)
## 4.1 merge each patches together
threshold = 20
mergelist=[]
# check => up and down
for i in range(4):
    for j in range(5):
        num,numm = (i*5+j),((i+1)*5+j)
        boxes,boxes_under = results[num]['rois'],results[numm]['rois']
        for m in range(boxes.shape[0]):
            for mm in range(boxes_under.shape[0]):
                if boxes[m][2]>=512 and boxes_under[mm][0]<=0:
                    if np.abs(boxes[m][1]-boxes_under[mm][1])<=threshold and np.abs(boxes[m][3]-boxes_under[mm][3])<=threshold:
                        mergelist.append([[num,m],[numm,mm]])
# check => left and right        
for i in range(5):
    for j in range(4):
        num,numm = (i*5+j),(i*5+(j+1))
        boxes,boxes_right = results[num]['rois'],results[numm]['rois']
        for m in range(boxes.shape[0]):
            for mm in range(boxes_right.shape[0]):
                if boxes[m][3]>=510 and boxes_right[mm][1]<=2:
                    if np.abs(boxes[m][0]-boxes_right[mm][0])<=threshold and np.abs(boxes[m][2]-boxes_right[mm][2])<=threshold:
                        mergelist.append([[num,m],[numm,mm]])
                        
checklist = np.array(mergelist).reshape(len(mergelist)*2,2)
print("checklist.shape: ", checklist.shape, flush=True)

total_boxes=[]
total_masks=[]
for num in range(25):
    i,j = num//5,num%5
    boxes,masks = results[num]['rois'],results[num]['masks']
    for N in range(boxes.shape[0]):
        ischeck=True
        for check in checklist:
            if [num,N]==[check[0],check[1]]:
                ischeck=False
        if ischeck:
            total_boxes.append([(boxes[N][0]+512*i),(boxes[N][1]+512*j),(boxes[N][2]+512*i),(boxes[N][3]+512*j)])

            big_mask = np.zeros((2560,2560), dtype=bool)
            big_mask[i*512:(i+1)*512,j*512:(j+1)*512] = masks[:,:,N]
            total_masks.append(big_mask)
            
print("len(total_boxes): ",len(total_boxes), flush=True)
print("len(total_masks): ",len(total_masks), flush=True)


if len(total_boxes)==0:
    print("len(total_boxes) is 0, so stop this job, the image is ", imagename[:-8])
else:
    for merge in mergelist:
        num = merge[0][0]
        i,j,N = num//5,num%5,merge[0][1]
        boxes,masks = results[num]['rois'],results[num]['masks']
        box = [(boxes[N][0]+512*i),(boxes[N][1]+512*j),(boxes[N][2]+512*i),(boxes[N][3]+512*j)]
        big_mask = np.zeros((2560,2560), dtype=bool)
        big_mask[i*512:(i+1)*512,j*512:(j+1)*512] = masks[:,:,N]

        numm = merge[1][0]
        ii,jj,NN = numm//5,numm%5,merge[1][1]
        bboxes,mmasks = results[numm]['rois'],results[numm]['masks']
        bbox = [(bboxes[NN][0]+512*ii),(bboxes[NN][1]+512*jj),(bboxes[NN][2]+512*ii),(bboxes[NN][3]+512*jj)]
        big_mask[ii*512:(ii+1)*512,jj*512:(jj+1)*512] = mmasks[:,:,NN]

        total_boxes.append([min(box[0],bbox[0]),min(box[1],bbox[1]),max(box[2],bbox[2]),max(box[3],bbox[3])])
        total_masks.append(big_mask)

    total_boxes=np.array(total_boxes)
    total_masks=np.transpose(np.array(total_masks),(1,2,0))
    total_class_ids=np.ones(total_boxes.shape[0], dtype=np.int32)
    print("total_boxes.shape: ",total_boxes.shape, flush=True)
    print("total_masks.shape: ",total_masks.shape, flush=True)

    ## 4.2 remove Outliers
    if total_boxes.shape[0]<=10:
        F_total_boxes,F_total_masks,F_total_class_ids = total_boxes,total_masks,total_class_ids
    else:
        total_masks_area=[]
        for i in range(total_boxes.shape[0]):
            total_masks_area.append(np.sum(total_masks[:,:,i]))
        print("len(total_masks_area): ",len(total_masks_area), flush=True)

        total_boxes_size=[]
        for i in range(total_boxes.shape[0]):
            box = total_boxes[i]
            total_boxes_size.append((box[2]-box[0])/(box[3]-box[1]))
        print("len(total_boxes_size): ",len(total_boxes_size), flush=True)

        masks_outliers = []
        masks_zscore = np.abs(stats.zscore(total_masks_area))
        for zs in masks_zscore:
            masks_outliers.append((zs<=2))

        boxes_outliers = []
        boxes_zscore = np.abs(stats.zscore(total_boxes_size))
        for zs in boxes_zscore:
            boxes_outliers.append((zs<=2))

        print("masks_outliers: ", flush=True)
        for c in range(len(total_masks_area)):
            if not masks_outliers[c]:
                print((masks_outliers[c], c, total_masks_area[c], masks_zscore[c]), flush=True)
        print("boxes_outliers: ", flush=True)     
        for c in range(len(total_boxes_size)):
            if not boxes_outliers[c]:
                print((boxes_outliers[c], c, total_boxes_size[c], boxes_zscore[c]), flush=True)

        outliers = np.multiply(masks_outliers,boxes_outliers)

        F_total_boxes=[]
        F_total_masks=[]
        for cc in range(len(outliers)):
            if outliers[cc]:
                F_total_boxes.append(total_boxes[cc])
                F_total_masks.append(total_masks[:,:,cc])

        F_total_boxes=np.array(F_total_boxes)
        F_total_masks=np.transpose(np.array(F_total_masks),(1,2,0))
        F_total_class_ids=np.ones(F_total_boxes.shape[0], dtype=np.int32)
        print("total_boxes.shape: ",F_total_boxes.shape, flush=True)
        print("total_masks.shape: ",F_total_masks.shape, flush=True)
    print("##########################################################", flush=True)


    # 5. Split Cell
    print("5. Split Cell", flush=True)
    for nn in range(F_total_boxes.shape[0]):
        bb = F_total_boxes[nn]
        mmask = F_total_masks[:,:,nn][bb[0]:bb[2], bb[1]:bb[3]]
        iimg = img[bb[0]:bb[2], bb[1]:bb[3]][:,:,0]

        cell = np.multiply(iimg, mmask)
        cell = cv2.merge([cell,cell,cell])

        savename=imagename[:-8]+"_"+str(nn)+'.tif'
        cv2.imwrite(savepath+"/"+savename, cell)
        print("save name with ", savepath+"/"+savename, flush=True)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", flush=True)

    visualize.display_instances_save(img, total_boxes, total_masks, total_class_ids, ["BG","nucleus"], 
                                     savename=sys.argv[5]+"/"+imagename[:-8]+".png", figsize=(40, 40))