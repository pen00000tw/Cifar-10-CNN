import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
print('version :' , torch.__version__)
print('cuda :' , torch.cuda.is_available())
print('cudnn :' , torch.backends.cudnn.enabled)

EPOCH = 10
BATCH_SIZE = 200
LR = 0.001
DOWNLOAD_data = False

train_data = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    transform=torchvision.transforms.ToTensor(), #改成torch可讀
    download=DOWNLOAD_data,
)
test_data = torchvision.datasets.CIFAR10(root='./data/',train=False,transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=2000,shuffle=False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # (3, 32, 32)
            nn.Conv2d(
                in_channels=3,              # 厚度
                out_channels=32,            # 捲出來的厚度
                kernel_size=5,              
                stride=1,                   # 每次移動幾步
                padding=2,                  
            ),                              # (32, 32, 32)
            nn.ReLU(),                      # 啟動函數
            nn.MaxPool2d(kernel_size=2),    # 池化(32, 16, 16)
        )
        self.conv2 = nn.Sequential(         # (32, 16, 16)
            nn.Conv2d(32, 64, 5, 1, 2),     # (64, 16, 16)
            nn.ReLU(),                      # 啟動函數
            nn.MaxPool2d(2),                # (64, 8, 8)
        )
        self.F = nn.Linear(64*8*8,1024)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(1024, 10)   # 全連結層

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           
        x = self.F(x)
        x = self.drop(x)
        output = self.out(x)
        return output


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
now = time.time()
for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            for (test_x,test_y) in test_loader:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                tmp = int(time.time() - now)
                minute = tmp / 60
                seconds = tmp % 60
                print('代數: %000d, 次數: %000d, 測試資料準確率: %2.2f, 使用時間: %d:%02d'%(epoch,step,accuracy * 100,minute,seconds))
                #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                break  #拿2000筆出來訓練就好