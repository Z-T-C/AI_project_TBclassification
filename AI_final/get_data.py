import os
import random
import csv
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import re
import pandas as pd
import torch
import torchvision.transforms.functional as transfunc
from typing import Sequence
import random
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 随机划分 index
testindex = [2, 8, 21, 24, 35, 53, 60, 65, 69, 112, 142, 152, 165, 169, 172, 176, 194, 195, 198, 210, 213, 216, 221, 228, 232, 268, 272, 282, 289, 292, 295, 305, 310, 318, 336, 339, 341, 345, 346, 366, 372, 383, 386, 392, 395, 401, 406, 415, 432, 436, 497, 511, 521, 523, 545, 560, 566, 595, 604, 610, 616, 629, 639, 647, 651]
valindex = [9, 11, 13, 40, 41, 42, 43, 45, 47, 55, 59, 63, 67, 68, 69, 74, 75, 79, 81, 102, 108, 116, 121, 126, 137, 139, 144, 145, 146, 152, 159, 160, 166, 167, 172, 179, 186, 189, 200, 202, 210, 215, 218, 227, 229, 231, 232, 239, 240, 241, 242, 246, 250, 255, 256, 260, 264, 273, 275, 279, 280, 281, 287, 292, 297, 300, 306, 310, 315, 319, 324, 328, 336, 347, 349, 357, 359, 363, 376, 383, 406, 411, 419, 420, 431, 432, 440, 443, 454, 455, 459, 460, 462, 463, 470, 472, 481, 487, 498, 499, 506, 508, 511, 534, 538, 545, 546, 553, 568, 569, 574, 577, 581, 586, 588]
# valfile = ['CHNCXR_0278_0.png', 'CHNCXR_0208_0.png', 'CHNCXR_0200_0.png', 'CHNCXR_0249_0.png', 'CHNCXR_0180_0.png', 'CHNCXR_0239_0.png', 'CHNCXR_0181_0.png', 'CHNCXR_0197_0.png', 'CHNCXR_0103_0.png', 'CHNCXR_0100_0.png', 'CHNCXR_0061_0.png', 'CHNCXR_0112_0.png', 'CHNCXR_0129_0.png', 'CHNCXR_0291_0.png', 'CHNCXR_0247_0.png', 'CHNCXR_0050_0.png', 'CHNCXR_0196_0.png', 'CHNCXR_0040_0.png', 'CHNCXR_0313_0.png', 'CHNCXR_0043_0.png', 'CHNCXR_0069_0.png', 'CHNCXR_0243_0.png', 'CHNCXR_0079_0.png', 'CHNCXR_0166_0.png', 'CHNCXR_0274_0.png', 'CHNCXR_0168_0.png', 'CHNCXR_0051_0.png', 'CHNCXR_0240_0.png', 'CHNCXR_0118_0.png', 'CHNCXR_0281_0.png', 'CHNCXR_0307_0.png', 'CHNCXR_0214_0.png', 'CHNCXR_0088_0.png', 'CHNCXR_0004_0.png', 'CHNCXR_0150_0.png', 'CHNCXR_0273_0.png', 'CHNCXR_0283_0.png', 'CHNCXR_0176_0.png', 'CHNCXR_0098_0.png', 'CHNCXR_0272_0.png', 'CHNCXR_0241_0.png', 'CHNCXR_0116_0.png', 'CHNCXR_0299_0.png', 'CHNCXR_0310_0.png', 'CHNCXR_0054_0.png', 'CHNCXR_0264_0.png', 'CHNCXR_0311_0.png', 'CHNCXR_0269_0.png', 'CHNCXR_0305_0.png', 'CHNCXR_0267_0.png', 'CHNCXR_0503_1.png', 'CHNCXR_0416_1.png', 'CHNCXR_0575_1.png', 'CHNCXR_0629_1.png', 'CHNCXR_0386_1.png', 'CHNCXR_0634_1.png', 'CHNCXR_0606_1.png', 'CHNCXR_0335_1.png', 'CHNCXR_0331_1.png', 'CHNCXR_0447_1.png', 'CHNCXR_0488_1.png', 'CHNCXR_0554_1.png', 'CHNCXR_0578_1.png', 'CHNCXR_0454_1.png', 'CHNCXR_0336_1.png', 'CHNCXR_0639_1.png', 'CHNCXR_0482_1.png', 'CHNCXR_0646_1.png', 'CHNCXR_0470_1.png', 'CHNCXR_0364_1.png', 'CHNCXR_0654_1.png', 'CHNCXR_0523_1.png', 'CHNCXR_0376_1.png', 'CHNCXR_0464_1.png', 'CHNCXR_0401_1.png', 'CHNCXR_0433_1.png', 'CHNCXR_0494_1.png', 'CHNCXR_0517_1.png', 'CHNCXR_0458_1.png', 'CHNCXR_0333_1.png', 'CHNCXR_0549_1.png', 'CHNCXR_0392_1.png', 'CHNCXR_0391_1.png', 'CHNCXR_0642_1.png', 'CHNCXR_0567_1.png', 'CHNCXR_0662_1.png', 'CHNCXR_0421_1.png', 'CHNCXR_0658_1.png', 'CHNCXR_0424_1.png', 'CHNCXR_0384_1.png', 'CHNCXR_0367_1.png', 'CHNCXR_0420_1.png', 'CHNCXR_0577_1.png', 'CHNCXR_0618_1.png', 'CHNCXR_0537_1.png', 'CHNCXR_0457_1.png', 'CHNCXR_0387_1.png', 'CHNCXR_0400_1.png', 'CHNCXR_0418_1.png', 'CHNCXR_0520_1.png', 'CHNCXR_0548_1.png', 'CHNCXR_0632_1.png', 'CHNCXR_0569_1.png', 'CHNCXR_0435_1.png']
# testfile = ['CHNCXR_0286_0.png', 'CHNCXR_0039_0.png', 'CHNCXR_0026_0.png', 'CHNCXR_0021_0.png', 'CHNCXR_0056_0.png', 'CHNCXR_0259_0.png', 'CHNCXR_0109_0.png', 'CHNCXR_0246_0.png', 'CHNCXR_0087_0.png', 'CHNCXR_0306_0.png', 'CHNCXR_0194_0.png', 'CHNCXR_0073_0.png', 'CHNCXR_0213_0.png', 'CHNCXR_0285_0.png', 'CHNCXR_0162_0.png', 'CHNCXR_0089_0.png', 'CHNCXR_0031_0.png', 'CHNCXR_0005_0.png', 'CHNCXR_0255_0.png', 'CHNCXR_0221_0.png', 'CHNCXR_0309_0.png', 'CHNCXR_0071_0.png', 'CHNCXR_0244_0.png', 'CHNCXR_0115_0.png', 'CHNCXR_0236_0.png', 'CHNCXR_0223_0.png', 'CHNCXR_0002_0.png', 'CHNCXR_0084_0.png', 'CHNCXR_0001_0.png', 'CHNCXR_0195_0.png', 'CHNCXR_0113_0.png', 'CHNCXR_0561_1.png', 'CHNCXR_0641_1.png', 'CHNCXR_0446_1.png', 'CHNCXR_0655_1.png', 'CHNCXR_0659_1.png', 'CHNCXR_0404_1.png', 'CHNCXR_0544_1.png', 'CHNCXR_0640_1.png', 'CHNCXR_0607_1.png', 'CHNCXR_0442_1.png', 'CHNCXR_0350_1.png', 'CHNCXR_0508_1.png', 'CHNCXR_0357_1.png', 'CHNCXR_0620_1.png', 'CHNCXR_0535_1.png', 'CHNCXR_0365_1.png', 'CHNCXR_0332_1.png', 'CHNCXR_0465_1.png', 'CHNCXR_0341_1.png', 'CHNCXR_0615_1.png', 'CHNCXR_0591_1.png', 'CHNCXR_0422_1.png', 'CHNCXR_0611_1.png', 'CHNCXR_0525_1.png', 'CHNCXR_0551_1.png', 'CHNCXR_0617_1.png', 'CHNCXR_0656_1.png', 'CHNCXR_0581_1.png', 'CHNCXR_0408_1.png', 'CHNCXR_0624_1.png', 'CHNCXR_0382_1.png', 'CHNCXR_0573_1.png', 'CHNCXR_0660_1.png', 'CHNCXR_0593_1.png', 'CHNCXR_0613_1.png']


def getfile(data=data):
    Test = []
    Train = []
    Val = []
    # testindex = random.sample(range(len(data)), len(data)//10)
    # train_val = [i for i in range(len(data)) if i not in testindex]
    # valindex = random.sample(range(len(train_val)), len(data)//10)
    # print(testindex)
    # print(valindex)
    for i in range(len(data)):
        name = data[i]
        if i in testindex:
            Test.append(name)
        elif i in valindex:
            Val.append(name)
        else:
            Train.append(name)
    return Train, Val, Test



def GetData(train=0):
    filename = '../images/data_list.csv'
    datalst = []
    textfile = './encoded_features.csv'
    df = pd.read_csv(textfile)

    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)  
        for row in csv_reader: 
            datalst.append(row[0])

    

    trainfile, valfile, testfile = getfile(data=datalst)
   
    data_path = []

    for name in datalst:
        # print(name)

        if train == 0:
            if (name in trainfile):
            # if name in valfile or name in testfile:
                continue
        elif train == 1:
            if (name not in valfile):
                continue
        elif train == 2:
            if (name not in testfile):
                continue
        
        textfeature = df.loc[df['study_id'] == name].iloc[:, 1:].values.flatten().tolist()
        

        match = re.search(r'_(\d+)\.png$', name)

        if match:
            label = int(match.group(1))
        else:
            raise ValueError(f"No matching pattern found in filename: {filename}")
        
        if train == 0:
            if label == 0:
                data_path.append(('./augmented_train/normal/' + name, textfeature[:2], textfeature[2:], int(label)))
            else:
                data_path.append(('./augmented_train/ill/' + name, textfeature[:2], textfeature[2:], int(label)))
        elif train == 1:
            if label == 0:
                data_path.append(('./augmented_val/normal/' + name, textfeature[:2], textfeature[2:], int(label)))
            else:
                data_path.append(('./augmented_val/ill/' + name, textfeature[:2], textfeature[2:], int(label)))
        elif train == 2:
            if label == 0:
                data_path.append(('./augmented_test/normal/' + name, textfeature[:2], textfeature[2:], int(label)))
            else:
                data_path.append(('./augmented_test/ill/' + name, textfeature[:2], textfeature[2:], int(label))) 
        
        # data_path.append(('../images/images/' + name, int(label)))
        
    return data_path


class MyDataset():
    def __init__(self, istrain, transform=None, target_transform=None):
        data = GetData(train=istrain)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        xpth, f1,f2, lab_y = self.data[index]  # 引入病例文本信息
        # xpth, lab_y = self.data[index]  # 仅图像
        if self.transform is not None:
            x = self.transform(Image.open(xpth).convert('L'))

        if self.target_transform is not None:
            lab_y = self.target_transform(lab_y)
            # lab_y = self.target_transform(lab_y).ToTensor()
        f1, f2 = torch.tensor(f1, dtype=torch.long), torch.tensor(f2, dtype=torch.long)
        # return x, lab_y
        return x, f1, f2, lab_y

    def __len__(self):
        return len(self.data)
    
class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transfunc.rotate(x, angle)


'''   
train_transforms = transforms.Compose([  
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        MyRotateTransform([-90,-45,0,45,90]) , # 使用旋转，对数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])


val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])

# 下面是调试代码
if __name__ == '__main__':
    filename = '../images/data_list.csv'
    datalst = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            datalst.append(row[0])   
    traindata = GetData(train=0)
    valdata = GetData(train=1)
    testdata = GetData(train=2)
'''

    
    