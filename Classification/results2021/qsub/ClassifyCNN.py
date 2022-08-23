import cv2,os
import sys
import numpy as np
from matplotlib import pyplot as plt

mask = sys.argv[1]
lens = sys.argv[2]
mmmodel = sys.argv[3]
print("This is the 1) mask:", mask, "; 2)Lens: ", lens, "; model: ",mmmodel)

if mask == "TRUE":
    mask = True
else:
    mask = False
    
# # データの読み込み
if mask == True:
    Data_02Nami=np.load("../imread_02Namix"+str(lens)+"_mask.npy",allow_pickle=True)
    Data_01=np.load("../imread_01x"+str(lens)+"_mask.npy",allow_pickle=True)
else:
    Data_02Nami=np.load("../imread_02Namix"+str(lens)+"_nomask.npy",allow_pickle=True)
    Data_01=np.load("../imread_01x"+str(lens)+"_nomask.npy",allow_pickle=True)
print("Data_01.shape:", Data_01.shape, flush=True)
print("Data_02Nami.shape:", Data_02Nami.shape, flush=True)


# # データ処理
# ## 1. Padding Unify the size
if int(lens) == 40:
    Data_01=Data_01[:40000]
    print("Data_01.shape:", Data_01.shape)
    print("Data_02Nami.shape:", Data_02Nami.shape)
    Max_shape_0 = 512
    Max_shape_1 = 512
elif int(lens) == 20:
    Max_shape_0 = 256
    Max_shape_1 = 256
    
#　## 2. 同じサイズにする 
def datapadding(data):
    DataPad=[]
    for img in data:
        imgSize = img.shape
        top_size,bottom_size = (Max_shape_0-imgSize[0])//2,(Max_shape_0-imgSize[0])//2
        left_size,right_size = (Max_shape_1-imgSize[1])//2,(Max_shape_1-imgSize[1])//2
        if (imgSize[0] % 2) != 0:
            top_size,bottom_size = (Max_shape_0-imgSize[0])//2,(Max_shape_0-imgSize[0])//2+1
        if (imgSize[1] % 2) != 0:     
            left_size,right_size = (Max_shape_1-imgSize[1])//2,(Max_shape_1-imgSize[1])//2+1
        img_pad = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=(0,0,0))
        DataPad.append(img_pad)
    return DataPad

DataPad_01 = datapadding(Data_01)
DataPad_02Nami = datapadding(Data_02Nami)
print("DataPad_01: ",len(DataPad_01))
print("DataPad_02Nami: ",len(DataPad_02Nami))

# ## 3. Split Train and Test
print("Split Train and Test", flush=True)
DataPad_01 = DataPad_01
DataLabel_01 = np.zeros(len(DataPad_01), dtype=np.int)
DataPad_02Nami = DataPad_02Nami
DataLabel_02Nami = np.ones(len(DataPad_02Nami), dtype=np.int)


from sklearn.model_selection import train_test_split
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(DataPad_01, DataLabel_01, test_size=0.3, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(DataPad_02Nami, DataLabel_02Nami, test_size=0.3, random_state=42)
X_train, X_test = np.concatenate((X_train_0, X_train_1), axis = 0), np.concatenate((X_test_0, X_test_1), axis = 0)
y_train, y_test = np.concatenate((y_train_0, y_train_1), axis = 0), np.concatenate((y_test_0, y_test_1), axis = 0)
print("Total number of train : ", len(y_train))
print("train_class_0 num : ", y_train.tolist().count(0))
print("train_class_1 num : ", y_train.tolist().count(1))
print("")
print("Total number of test : ", len(y_test))
print("test_class_0 num : ", y_test.tolist().count(0))
print("test_class_1 num : ", y_test.tolist().count(1))
print("")

# # データ前処理
print("Data Processing ", flush=True)
import torch
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
torch.__version__

class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        data = x_train.astype('float32')
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.y_train = y_train
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.x_train)
    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(y_train[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test, y_test):
        data = x_test.astype('float32')
        self.x_test = []
        for i in range(data.shape[0]):
            self.x_test.append(Image.fromarray(np.uint8(data[i])))
        self.y_test = y_test
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.x_test)
    def __getitem__(self, idx):
        return self.transform(self.x_test[idx]), torch.tensor(y_test[idx], dtype=torch.long)

trainval_data = train_dataset(X_train, y_train)
test_data = test_dataset(X_test, y_test)


batch_size = 64
val_size = int(len(trainval_data)*0.2)
train_size = len(trainval_data) - val_size
print("train_size: ",train_size, flush=True)
print("val_size: ",val_size, flush=True)
print("test_size: ",len(y_test), flush=True)
print("")
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])
dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)


# # ResNet遷移学習
import torch
import torchvision.models as models
import torch.nn as nn

if mmmodel == "Resnet101": 
    model = models.resnet101(pretrained=True)
    #Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
elif mmmodel == "Resnet18": 
    model = models.resnet18(pretrained=True)
    #Freeze model weights
    for param in model.parameters():
        param.requires_grad = True
        
num_fcs = model.fc.in_features
# FC層のクラス数を変更
model.fc = nn.Sequential(
    nn.Linear(num_fcs, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 2)
)
for param in model.fc.parameters():
    param.requires_grad = True
model.avgpool = nn.AdaptiveAvgPool2d(1)
loss_func = nn.NLLLoss()

# model = model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#第一行代码
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)#第二行代码

print("Start Training", flush=True)
n_epochs = 200
# weight_for_0 : 1. / negative * (negative + positive) negative : ラベル0の数
# weight_for_1 : 1. / positive * (negative + positive) positive : ラベル1の数
# class_weight = {0 : weight_for_0, 1 : weight_for_1}
weights = torch.tensor([(len(DataPad_01)+len(DataPad_02Nami))/len(DataPad_01), 
                        (len(DataPad_01)+len(DataPad_02Nami))/len(DataPad_02Nami)]).cuda()
loss_function = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # 定义衰减策略
# device = 'cuda'

losstrain=[]
lossvalid=[]
Accuracytrain=[]
Accuracyvalid=[]

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    # Train
    optimizer.step()
    model.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        n_train += t.size()[0]
        model.zero_grad()  # 勾配の初期化
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = model.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        loss.backward()  # 誤差の逆伝播
        optimizer.step()  # パラメータの更新
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())
    # Evaluate
    model.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        n_val += t.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = model.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())
    scheduler.step()
    # Visualize loss & accuracy    
    losstrain.append(np.mean(losses_train))   
    Accuracytrain.append(acc_train/n_train)
    lossvalid.append(np.mean(losses_train))
    Accuracyvalid.append(acc_val/n_val)
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy:{:.3f}]'.format(epoch,np.mean(losses_train),acc_train/n_train,np.mean(losses_valid),acc_val/n_val), flush=True)


# train processing plot, y axis is from 0 to 1.0
epochs=range(1,n_epochs+1)
plt.ylim(0,1.0)
plt.plot(epochs,Accuracytrain,'b',label='Training accuracy')  
plt.plot(epochs, Accuracyvalid,'r',label='Validation accuracy')
plt.title('Training and Validation accuracy')
name=sys.argv[1]+sys.argv[2]+sys.argv[3]+"_Output.jpg"
plt.savefig(name)
print("Saved output as , ", name, flush=True)
print("")


# train processing plot, y axis is from 0.6 to 1.0
plt.ylim(0.6,1.0)
plt.plot(epochs,Accuracytrain,'b',label='Training accuracy')  
plt.plot(epochs, Accuracyvalid,'r',label='Validation accuracy')
plt.title('Training and Validation accuracy')
name_specify=sys.argv[1]+sys.argv[2]+sys.argv[3]+"_Output_specify.jpg"
plt.savefig(name_specify)
print("Saved output as , ", name_specify, flush=True)
print("")

# save model
PATH = "Model_"+sys.argv[1]+sys.argv[2]+sys.argv[3]+".pkl"
torch.save(model, PATH)


print("TEST : ")
losses_test = []
n_test = 0
acc_test = 0
model.eval()

for x, t in dataloader_test:
    n_test += t.size()[0]
    x = x.to(device)  # テンソルをGPUに移動
    t = t.to(device)
    y = model.forward(x)  # 順伝播
    loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
    pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
    acc_test += (pred == t).float().sum().item()
    losses_test.append(loss.tolist())

# Visualize loss & accuracy 
print("acc_test: ", acc_test) 
print("n_test: ", n_test)
print('Loss: {:.3f}, Accuracy: {:.3f}'.format(np.mean(losses_test),acc_test/n_test), flush=True)