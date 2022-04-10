import warnings
warnings.filterwarnings("ignore")

import cv2
import os
import shutil
import sys
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label, regionprops
import tensorflow
import keras
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from imgaug import augmenters as iaa
from datetime import timedelta
import datetime
import imageio
import math
import pytz
from pytz import timezone
import imagecodecs._imcd
from tifffile import imread


# Setup DataPath, Savepath and Process Images
DATA_PATH = "./root/datasets/jupyter"
TEST_PATH = DATA_PATH+"/test/"
TEST_LABEL_PATH = DATA_PATH+"/testlabel"   
NOISE_IMG_PATH = DATA_PATH+"/test"
IMG_SAVE_PATH = DATA_PATH+"/imagesDN"
NOISE_MAP = DATA_PATH+"/Noise/map" 

# Setup DataPath, Savepath and Process Images

imagepath=sys.argv[1]
imagename=sys.argv[2]
savepath=sys.argv[3]

print("imagepath is ",imagepath, flush=True)
print("imagename is ",imagename, flush=True)
print("savepath is ",savepath, flush=True)

DATA_PATH = "./root/datasets/"+imagename[:-8]
TEST_PATH = DATA_PATH+"/test/"
TEST_LABEL_PATH = DATA_PATH+"/testlabel"   
NOISE_IMG_PATH = DATA_PATH+"/test"
IMG_SAVE_PATH = DATA_PATH+"/imagesDN"
NOISE_MAP = DATA_PATH+"/Noise/map" 

if os.path.exists(DATA_PATH) and os.path.isdir(DATA_PATH):
    shutil.rmtree(DATA_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(TEST_LABEL_PATH):
    os.makedirs(TEST_LABEL_PATH)
if not os.path.exists(NOISE_IMG_PATH):
    os.makedirs(NOISE_IMG_PATH)
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)
if not os.path.exists(NOISE_MAP):
    os.makedirs(NOISE_MAP)


# 前処理
print("##########################################################")
print(">>>>  Read image and processing", flush=True)
def gamma_img(gamma, img):
    gamma_cvt = np.zeros((256,1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
    return cv2.LUT(img, gamma_cvt)
img = cv2.imread(imagepath)
img_gamma = gamma_img(1, img)
savetest = TEST_PATH+imagename
cv2.imwrite(savetest, img_gamma)

M=512
N=512
NME = "Test"
Resize = False
Norm = False
Noise = False
Split = True
Write = True
Spath = DATA_PATH
PATH = TEST_PATH
for name in os.listdir(PATH):
    print("Spath: ",Spath, flush=True)
    print("PATH: ",PATH, flush=True)
    print("name:", name[:len(name)-8], flush=True)
    if Split == True:
        new_shape=[M,N]
        top_size,bottom_size,left_size,right_size=0,0,0,0
        d,c = new_shape[0],img.shape[0]
        dd,cc = new_shape[1],img.shape[1]
        if c//d != c/d or cc//dd != cc/dd:
            if c//d != c/d:
                top_size,bottom_size = int(((c//d+1)*d-c)/2),int(((c//d+1)*d-c)/2)
            if cc//dd != cc/dd:
                left_size,right_size=int(((cc//dd+1)*d-cc)/2),int(((cc//dd+1)*dd-cc)/2)
            img=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=(0,0,0))
        x=0
        for i in range(int(img.shape[0]/512)):
            for j in range(int(img.shape[0]/512)):
                imgsplit=img[i*512:(i+1)*512:,j*512:(j+1)*512,:]
                savename=name[:len(name)-8]+"_"+str(x)+".tif"
                cv2.imwrite(PATH+name[:len(name)-8]+"_"+str(x)+".tif",imgsplit)
                x+=1
    os.remove(savetest)
    
    
# Run this cell to conduct processing specified above.
MP = 0
Num = 0
fin = 0

Spath = DATA_PATH
PATH = TEST_PATH
while fin == 0:
    for name in os.listdir(PATH):
        path = os.path.join(PATH, name)
        path2 = os.path.join(TEST_LABEL_PATH, name)
        if os.path.isdir(path):
            continue
        if os.path.isdir(path2):
            continue
        ll = len(name)
        print(name)
        # Get Extension
        if Num == 0:
            nme, ext = os.path.splitext(name)
        if ext == '.tif' or ext == '.tiff':
            img = cv2.imread(path)
            R,G,B = cv2.split(img)
            img = R
        else:
            img = cv2.imread(path,0)
        img = img.astype('uint8') 
        sh = img.shape
        # Pixel Average
        if len(sh) == 3 or 4:
            MP = (np.mean(img,axis=(0,1)) + MP)
        else:
            MP = (np.mean(img) + MP)
        Num = Num + 1
        if Write == True:
            if not os.path.exists(os.path.join(Spath, NME, name[0:ll-4], "images")):
                os.makedirs(os.path.join(Spath, NME, name[0:ll-4], "images"))
            cv2.imwrite(os.path.join(Spath, NME, name[0:ll-4], "images", name[0:ll-3]+'png'), img)
        if len(os.listdir(TEST_LABEL_PATH))!=0:
            if ext == '.tif' or ext == '.tiff':
                img2 = imread(path2,0)
            else:
                # img2 = cv2.imread(path2)
                img2 = cv2.cvtColor(path2,cv2.COLOR_BGR2GRAY)
            img2 = label(img2)
            img2 = cv2.resize(img2,(M,N),interpolation=cv2.INTER_NEAREST)
            P = img2.max()
            out = np.zeros([M, N, P])
            if Write == True:
                if not os.path.exists(os.path.join(Spath, NME, name[0:ll-4], "masks")):
                    os.mkdir(os.path.join(Spath, NME, name[0:ll-4], "masks"))
                for n in range(1,P+1):
                    ind = np.where(img2 == n)
                    for i in range(0,ind[0].shape[0]-1):
                        out[ind[0][i],ind[1][i],n-1] = 255
                    cv2.imwrite(os.path.join(Spath, NME, name[0:ll-4], "masks", name[0:ll-4] + "_" + str(n-1) + ".png"),out[:,:,n-1])
        elif len(os.listdir(TEST_LABEL_PATH))==0:
            img2 = np.ones([M,N])
            out = np.zeros([M, N, 1])
            if Write == True:
                if not os.path.exists(os.path.join(Spath, NME, name[0:ll-4], "masks")):
                    os.mkdir(os.path.join(Spath, NME, name[0:ll-4], "masks"))
            for n in range(1,2):
                ind = np.where(img2 == n)
                for i in range(0,ind[0].shape[0]-1):
                    out[ind[0][i],ind[1][i],n-1] = 255
                cv2.imwrite(os.path.join(Spath, NME, name[0:ll-4], "masks", name[0:ll-4] + "_" + str(n-1) + ".png"),out[:,:,n-1])
        fin = 1

MP = MP/Num
if len(sh) != 3 or 4:
    MP2 = np.array([MP,MP,MP])
else:
    MP2 = MP
    
    
# Configuration
DETECTION_MIN_CONFIDENCE = 0.9
DETECTION_NMS_THRESHOLD = 0.5
RPN_NMS_THRESHOLD = 0.5
sys.path.append("./root/Mask_RCNN/")
import colorspacious
from glasbey import Glasbey
from skimage.color import label2rgb
import math
class NucleusConfig(Config):
    NAME = "nucleus"
    IMAGES_PER_GPU = 1  #@param {type:"integer"}
    NUM_CLASSES = 2  #@param {type:"integer"}
    EPOCHS =  100#@param {type:"integer"}
    STEPS_PER_EPOCH = 100 #@param {type:"integer"}
    VALIDATION_STEPS =  20#@param {type:"integer"}
    DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE #@param {type:"number"}
    DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD #@param {type:"number"}
    RPN_NMS_THRESHOLD = RPN_NMS_THRESHOLD #@param {type:"number"}
    BACKBONE = "resnet50" #@param ["resnet50", "resnet101"] {type:"string"} 
    IMAGE_RESIZE_MODE = "pad64" #@param ["none", "crop", "square", "pad64"] {type:"string"}
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    MEAN_PIXEL = MP2
    USE_MINI_MASK = False #@param {type:"boolean"}
    height = 128 #@param {type:"integer"}
    width = 128 #@param {type:"integer"}
    MINI_MASK_SHAPE = (height, width)  # (height, width) of the mini-mask
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 256
    LEARNING_RATE = 0.0001 #@param {type:"number"}
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.00001 #@param {type:"number"}

class NucleusInferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = RPN_NMS_THRESHOLD

config = NucleusConfig()

# Define Functions
class NucleusDataset(utils.Dataset):
    def load_nucleus(self, dataset_dir, subset):
        DSET_NAMES = "nucleus" #@param {type: "string"}
        CLASS_NAME = "nucleus" #@param {type: "string"}
        self.add_class("nucleus", 1, "nucleus")
        dataset_dir = os.path.join(dataset_dir, subset)
        image_ids = next(os.walk(dataset_dir))[1]
        print("dataset_dir is ", dataset_dir)
        print("image_ids is ", image_ids)

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))


# Detection Module
# Conduct post-processing on predictions?
PROC = True #@param {type:"boolean"}
utc = pytz.utc
utc_dt = datetime.datetime.now()
eastern = timezone('US/Eastern')
loc_dt = utc_dt.astimezone(eastern)


color = np.array(([1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,0.5,0],[0.5,1,0],[0,1,0.5],[0,0.5,1],[1,0,0.5],[0.5,0,1],[1,0.5,0.25],[0.25,0.5,1],[1,0.25,0.5],[0.5,0.25,1],[0.5,1,0.25],[0.25,1,0.5]),np.float32)
gb = Glasbey(base_palette=color, chroma_range = (60,100), no_black=True)
c4 = gb.generate_palette(size=18)
color4 = c4[1:]

def normalized(rgb):
        norm=np.zeros((512,512,3),np.float32)
        norm_rgb=np.zeros((512,512,3),np.uint8)
        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]
        sum=b+g+r
        norm[:,:,0]=b/sum*255.0
        norm[:,:,1]=g/sum*255.0
        norm[:,:,2]=r/sum*255.0
        norm_rgb=cv2.convertScaleAbs(norm)
        return norm_rgb

def overlay(mask, orig, clr):
    maskPR = label(mask)
    labels = label2rgb(label=maskPR, bg_label=0, bg_color=(0, 0, 0), colors=clr)
    L2 = normalized(labels)
    if len(orig.shape) < 3: 
        O2 = cv2.cvtColor(orig.astype('uint8'), cv2.COLOR_GRAY2BGR)
    else:
        O2 = orig
    comb = cv2.addWeighted(L2.astype('float64'),0.5,O2.astype('float64'),0.5,0)
    return comb


# Results Output Folder Name
# * submit_dir = 'Tissue_Nucleus'  + '_' +  loc_dt.strftime('%Y-%m-%d_%H:%M:%S_%Z%z')
def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR) 
    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    ### b. Results Output Folder Name
    #submit_dir = 'Tissue_Nucleus'  + '_' +  loc_dt.strftime('%Y-%m-%d_%H:%M:%S_%Z%z')
    submit_dir = "Result"+loc_dt.strftime('%Y-%m-%d_%H:%M:%S_%Z%z')#@param {type: "raw"} 
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    mask_dir = submit_dir + "/masks"
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    rois_dir = submit_dir + "/rois"
    if not os.path.exists(rois_dir):
        os.makedirs(rois_dir)
    # Load over images
    init = 0
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        np.save(submit_dir + '/rois/'+ source_id+"_boxes.npy",r['rois'])
        np.save(submit_dir + '/rois/'+ source_id+"_masks.npy",r["masks"])
        submission.append(rle)
        # Save image with masks
        print(source_id)
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions", captions = None)
    print("Saved to ", submit_dir)
    return submit_dir,r


# ### RLE Encoding and Decoding

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

sys.path.append("../../")


# # Run the Network
Weights = "Kaggle" 
#Change weight paths if different.
if Weights == "coco":
    weights_path = '../../Datasets/weights/mask_rcnn_coco.h5' 
elif Weights == "imagenet":
    weights_path = model.get_imagenet_weights()
elif Weights == "Kaggle":
    weights_path = '../../Datasets/weights/mask_rcnn_kaggle_v1.h5' 
elif Weights == "Storm_Tissue":
    weights_path = '../../Datasets/weights/mask_rcnn_nucleus_tissue.h5' 
elif Weights == "Storm_Cell":
    weights_path = '../../Datasets/weights/mask_rcnn_nucleus_cell.h5' 
elif Weights == "Select":
    weights_path = '../../Datasets/weights/select_weight.h5' 


# ## Select Segmentation Results Directory
RESULTS_DIR = './root/results' 
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# Run Network
print("##########################################################")
print("Run Network", flush=True)
# Configurations
config = NucleusInferenceConfig()
# Create model
model = modellib.MaskRCNN(mode="inference", config=config,model_dir=weights_path)
if Weights == "coco":
    model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)
sdir,r = detect(model, DATA_PATH, subset='Test')


#####################################################################################


# Split Rois
# read image and padding
print("##########################################################")
print(">>>>  Split Rois", flush=True)
# read image and padding
img = cv2.imread(imagepath)
img = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=(0,0,0))
c=0
for i in range(int(img.shape[0]/512)):
    for j in range(int(img.shape[0]/512)):
        imgsplit=img[i*512:(i+1)*512, j*512:(j+1)*512, :]
        
        ######################################
        name = imagename[:-8]+"_"+str(c)
        c+=1
        boxes = np.load(sdir+"/rois/"+name+"_boxes.npy",allow_pickle=True)
        masks = np.load(sdir+"/rois/"+name+"_masks.npy",allow_pickle=True)
        if boxes.shape[0]==masks.shape[2]!=0:
            print(c, " boxes.shape",boxes.shape, flush=True)
            print(c, " masks.shape",masks.shape, flush=True)
            img_mask_all = []
            for b in range(masks.shape[2]):
                mask = masks[:,:,b]
                img_mask = np.zeros((512,512))
                for x in range(512):
                    for y in range(512):
                        if mask[x][y] == True:
                            img_mask[x][y]=1
                img_mask_all.append(img_mask)
            img_mask_all = np.array(img_mask_all)
            
            #####################################
            for a in range(boxes.shape[0]):
                pic = imgsplit[boxes[a][0]:boxes[a][2], boxes[a][1]:boxes[a][3]]
                mask = img_mask_all[a,:,:]
                mask = mask[boxes[a][0]:boxes[a][2], boxes[a][1]:boxes[a][3]]
                mask = cv2.merge([mask,mask,mask])
                pic_mask=np.multiply(pic,mask)
                pic_mask=pic_mask.astype(np.uint8)
                # imwrite
                savename=imagename[:-8]+"_"+str(c)+"_"+str(a)+'.tif'
                cv2.imwrite(savepath+"/"+savename, pic_mask)
                print("save name with ", savepath+"/"+savename, flush=True)
print("##########################################################")



