import cv2,os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from collections import OrderedDict
import math

# model load
print("load model", flush=True)
PATH = "../qsub220208/Model_Resnet18_nomask.pkl"
model = torch.load(PATH)
device = "cuda"

Max_shape_0=256
Max_shape_1=256
# 定义钩子函数，获取指定层名称的特征
print("define each function", flush=True)
feature_activation = {} # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        feature_activation[name] = output.detach()
    return hook


def dataTransform(img_path):
    img = cv2.imread(img_path)
    imgSize = img.shape
    top_size,bottom_size = (Max_shape_0-imgSize[0])//2,(Max_shape_0-imgSize[0])//2
    left_size,right_size = (Max_shape_1-imgSize[1])//2,(Max_shape_1-imgSize[1])//2
    if (imgSize[0] % 2) != 0:
        top_size,bottom_size = (Max_shape_0-imgSize[0])//2,(Max_shape_0-imgSize[0])//2+1
    if (imgSize[1] % 2) != 0:     
        left_size,right_size = (Max_shape_1-imgSize[1])//2,(Max_shape_1-imgSize[1])//2+1
    imgpad = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=(0,0,0))


    # 将图片处理成模型可以预测的形式
    transform = transforms.Compose([transforms.ToTensor()])
    input_img = transform(imgpad).unsqueeze(0).to(device)
    return imgpad,input_img,imgSize,top_size,bottom_size,left_size,right_size


def getfeature(img_path,plt):
    imgpad,input_img,imgSize,top_size,bottom_size,left_size,right_size = dataTransform(img_path)
    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))
    output = model(input_img)
    feature_activation.pop("module.avgpool")
    feature_activation.pop("module.fc.0")
    feature_activation.pop("module.fc.1")
    feature_activation.pop("module.fc.2")
    feature_activation.pop("module.fc.3")
    feature_activation.pop("module.fc.4")
    feature_activation.pop("module.fc")
    feature_activation.pop("module")
    feature_activation.pop("")
    if plt == True:
        # subplt each feature matrix
        for key in feature_activation:
            bn = feature_activation[key].cpu()
            print(key," : ",bn.shape)
            s = int(imgpad.shape[0]/bn.shape[2])
            n = math.ceil(math.sqrt(bn.shape[1]))
            plt.figure(figsize=(20,20))
            for i in range(bn.shape[1]):
                plt.subplot(n,n,i+1)
                plt.imshow(bn[0,i,
                                  int(top_size/s):int((top_size+imgSize[0])/s),
                                  int(left_size/s):int((left_size+imgSize[1])/s)], cmap='gray')
                plt.axis('off')
            plt.show()
            
            
            
# Save all feature map
print(">>>> Start saving", flush=True)
imgdir0120 = "../../../Datasets/211202NDAcquisition/CellsNoMask/NDAcquisition-01/"
imgdir0220 = "../../../Datasets/211202NDAcquisition/CellsNoMask/NDAcquisition-02Nami_x20/"
imgdir0140 = "../../../Datasets/211202NDAcquisition/CellsNoMask/NDAcquisition-01x40/"
imgdir0240 = "../../../Datasets/211202NDAcquisition/CellsNoMask/NDAcquisition-02Nami_x40/"

imgnamelist = ["NDAcquisition-01_XY001_1","NDAcquisition-01_XY001_8",
           "NDAcquisition-01x40_XY0001_1","NDAcquisition-01x40_XY0001_11",
           "NDAcquisition-02Nami_x20_XY009_1","NDAcquisition-02Nami_x20_XY136_2",
           "NDAcquisition-02Nami_x40_XY001_2","NDAcquisition-02Nami_x40_XY127_2"]
imgdirlist = [imgdir0120,imgdir0120,imgdir0140,imgdir0140,imgdir0220,imgdir0220,imgdir0240,imgdir0240]

for i in range(4,8):
    imgdir = imgdirlist[i]
    imgname = imgnamelist[i]
    img_path = imgdir + imgname + ".tif"
    print(">> Image : ", imgname, flush=True)
    getfeature(img_path,False)
    imgpad,input_img,imgSize,top_size,bottom_size,left_size,right_size = dataTransform(img_path)
    for key in feature_activation:
        bn = feature_activation[key].cpu()
        bn = np.array(bn)
        s = int(imgpad.shape[0]/bn.shape[2])
        n = math.ceil(math.sqrt(bn.shape[1]))
        plt.figure(figsize=(20,20))
        
        # save path
        path = "../Feature18/" + imgname + "/" + str(key)
        if not os.path.exists(path):
            os.makedirs(path)
            
        for i in range(bn.shape[1]):
            # save each feature in each layers
            savename = path+"/"+str(key)+"_"+str(i)+".png"
            plt.imsave(savename, bn[0,i,int(top_size/s):int((top_size+imgSize[0])/s),
                                  int(left_size/s):int((left_size+imgSize[1])/s)])
        
            # save all feature in each layers
            plt.subplot(n,n,i+1)
            plt.imshow(bn[0,i,int(top_size/s):int((top_size+imgSize[0])/s),
                              int(left_size/s):int((left_size+imgSize[1])/s)], cmap='gray')
        savename = path+"/"+str(key)+".png"
        plt.savefig(savename)
        print("Successfully saved ", savename, flush=True)





