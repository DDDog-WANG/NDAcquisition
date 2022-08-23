import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

def hiii():
    print("hi!!!!!")
    
    
class OriginIMG:
    def __init__(self,img):
        self.img = img
        self.shape = img.shape

    def pad(self,Max_shape_0,Max_shape_1):
        self.top,self.bottom = (Max_shape_0-self.shape[0])//2,(Max_shape_0-self.shape[0])//2
        self.left,self.right = (Max_shape_1-self.shape[1])//2,(Max_shape_1-self.shape[1])//2
        if (self.shape[0] % 2) != 0:
            self.top,self.bottom = (Max_shape_0-self.shape[0])//2,(Max_shape_0-self.shape[0])//2+1
        if (self.shape[1] % 2) != 0:     
            self.left,self.right = (Max_shape_1-self.shape[1])//2,(Max_shape_1-self.shape[1])//2+1
        imgpad = cv2.copyMakeBorder(self.img,self.top,self.bottom,self.left,self.right,
                                    cv2.BORDER_CONSTANT,value=(0,0,0))
        return imgpad

    def bin_ndarray(self,new_shape,operation):
        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if self.img.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(self.shape,new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape,self.shape)]
        flattened = [l for p in compression_pairs for l in p]
        imgresize = self.img.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(imgresize, operation)
            imgresize = op(-1*(i+1))
        return imgresize
    
    def predict(self,model,input_img,label):
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(input_img).unsqueeze(0).to(device)
        output = model(input_tensor)
        y = output.argmax(1).cpu().item()
        print("Prediction label: ", y==label)
        print("Real: ",label,";   Predict: ", y)
        
    def plot_feature(self,model,target_layer,input_img):
        
        for name, layer in model.named_modules():
            layer.register_forward_hook(get_activation(name))
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(input_img).unsqueeze(0).to(device)
        output = model(input_tensor)
        # plot each layer result
        for key in [target_layer]:
            bn = feature_activation[key].cpu()
            print(key," : ",bn.shape)
            s = int(input_img.shape[0]/bn.shape[2])
            n = math.ceil(math.sqrt(bn.shape[1]))
            plt.figure(figsize=(20,20))
            for i in range(bn.shape[1]):
                plt.subplot(n,n,i+1)
                plt.imshow(bn[0,i,int(self.top/s):int((self.top+self.shape[0])/s),
                                  int(self.left/s):int((self.left+self.shape[1])/s)], cmap='gray')
            plt.show()