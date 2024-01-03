import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import autograd, optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
from models.ResNet import resnet50
# from models.EfficientNet import efficientnet_b0
# import torchvision.models as models

import torchvision.transforms.functional as transfunc
from typing import Sequence
import random



def training(model, model_name, num_epochs, train_dataloader, val_dataloader):
    loss_function = torch.nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
   
    best_model = None
    # 添加最大准确度判断
    best_valacc = 0

    for epoch in tqdm(range(num_epochs)):

        print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_lr()))

        for phase in ['train', 'val']:

            epoch_loss = 0
            epoch_correct_items = 0
            epoch_items = 0

            if phase == 'train':
                
                model.train()
                with torch.enable_grad():
                    for _, data in enumerate(train_dataloader):
                        samples, targets = data[0].to(device), data[1].to(device)
                        optimizer.zero_grad()
                        outputs = model(samples)
                        loss = loss_function(outputs, targets)
                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets) 

                train_loss_array.append(epoch_loss / epoch_items)
                print('train_%d\'s loss is: %.3f%%' % (epoch, (100 * epoch_loss / epoch_items)))
                train_acc_array.append(epoch_correct_items / epoch_items)
                print('train_%d\'s acc is: %.3f%%' % (epoch, (100 * epoch_correct_items / epoch_items)))

                scheduler.step()

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(val_dataloader):
                        samples, targets = data[0].cuda(), data[1].cuda()
                        outputs = model(samples)
                        loss = loss_function(outputs, targets)
                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                val_loss_array.append(epoch_loss / epoch_items)
                print('val_%d\'s loss is: %.3f%%' % (epoch, (100 * epoch_loss / epoch_items)))
                val_acc_array.append(epoch_correct_items / epoch_items)
                print('val_%d\'s acc is: %.3f%%' % (epoch, (100 * epoch_correct_items / epoch_items)))

                if epoch_correct_items / epoch_items > best_valacc: # 以最大准确度为判断依据
                    best_valacc = epoch_correct_items / epoch_items
                    # torch.save(model.state_dict(), './results/trained_model/best_{}.pth'.format(model_name))
                    # best_model = copy.deepcopy(model)

                    #使用copy.deepcopy()直接深度拷贝训练中的model用来做validation是比较简洁的写法
                    print("\t| New best val acc for {}: {}".format(model_name, best_valacc))

    torch.save(model.state_dict(), './results/trained_model/{}.pth'.format(model_name))
    return best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array
# 不返回最优模型


# 可视化训练结果 每一轮的 训练损失 和精度
def visualize_training_results(train_loss_array,
                               val_loss_array,
                               train_acc_array,
                               val_acc_array,
                               num_epochs,
                               model_name,
                               batch_size):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("{} training | Batch size: {}".format(model_name, batch_size), fontsize=16)
    axs[0].plot(list(range(1, num_epochs + 1)), train_loss_array, label="train_loss")
    axs[0].plot(list(range(1, num_epochs + 1)), val_loss_array, label="val_loss")
    axs[0].legend(loc='best')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[1].plot(list(range(1, num_epochs + 1)), train_acc_array, label="train_acc")
    axs[1].plot(list(range(1, num_epochs + 1)), val_acc_array, label="val_acc")
    axs[1].legend(loc='best')
    axs[1].set(xlabel='epochs', ylabel='accuracy')
    plt.show()
    plt.savefig("./ResNet50_aug_loss_acc.jpg")



class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transfunc.rotate(x, angle)
    
train_transforms = transforms.Compose([  
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        MyRotateTransform([-90,-45,0,45,90]) , # 使用旋转，对数据增强
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])


val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])

if __name__ == '__main__':

    plt.switch_backend('agg')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device} device")

    batch_size = 16
    num_epochs = 200

    model = resnet50(num_classes=2)
    # model = efficientnet_b0(num_classes=2)
    # premodel = torch.load('../pretrained_models/resnet50-0676ba61.pth')
    # model.load_state_dict(premodel, strict=False)
    model = model.to(device)


    # 使用增强的数据集
    train_data = datasets.ImageFolder(root='./augmented_train', transform=train_transforms)
    train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4, prefetch_factor=8)
    # 同上
    val_data = datasets.ImageFolder(root='./augmented_val', transform=val_transforms)
    val_dataloader = DataLoader(val_data,batch_size=batch_size,shuffle=True,num_workers=4, prefetch_factor=8)
    '''
    train_dataset = MyDataset(istrain=0,transform=train_transforms)
    print('训练集长度:', len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=8)

    val_dataset = MyDataset(istrain=1,transform=val_transforms)
    print('验证集长度:', len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=8)
    '''
    
    _, train_loss_array, train_acc_array, val_loss_array, val_acc_array = training(model, "ResNet50_aug", num_epochs, train_dataloader, val_dataloader)


    visualize_training_results(train_loss_array, val_loss_array, train_acc_array, val_acc_array, num_epochs, "ResNet50_aug", batch_size)

