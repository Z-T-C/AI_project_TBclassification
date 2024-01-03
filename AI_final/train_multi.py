import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler

from models.ResNet import Multimodal
from get_data import MyDataset

import torchvision.transforms.functional as transfunc
from typing import Sequence
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(model, model_name, num_epochs, train_dataloader, val_dataloader):
    loss_function = torch.nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    train_loss_array, train_acc_array, val_loss_array, val_acc_array = [], [], [], []
    best_valacc = 0
    best_model = None

    for epoch in tqdm(range(num_epochs)):
        print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_lr()))

        for phase in ['train', 'val']:
            epoch_loss, epoch_correct_items, epoch_items = 0,0,0,0

            if phase == 'train':
                
                model.train()
                with torch.enable_grad():
                    for x,f1,f2, targets in train_dataloader:
                        x, f1, f2, targets= x.to(device), f1.to(device), f2.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(x,f1,f2)
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
                    for x,f1,f2, targets in val_dataloader:
                        x, f1, f2, targets= x.to(device), f1.to(device), f2.to(device), targets.to(device)
                        outputs = model(x,f1,f2)
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
                    #  torch.save(model.state_dict(), './results/trained_model/{}.pth'.format(model_name))
                    # best_model = copy.deepcopy(model)

                    print("\t| New best val acc for {}: {}".format(model_name, best_valacc))

    torch.save(model.state_dict(), './results/trained_model/{}.pth'.format(model_name))
    return best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array



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
    plt.savefig("./MyModel_loss_acc.jpg")




class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transfunc.rotate(x, angle)
    
train_transforms = transforms.Compose([  
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        MyRotateTransform([-90,-45,0,45,90]) , # 使用旋转，对数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])


val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])

if __name__ == '__main__':

    plt.switch_backend('agg')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    num_epochs = 200

    model = Multimodal()
    model = model.to(device)

    train_data = MyDataset(istrain=0, transform=train_transforms)
    train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4, prefetch_factor=8)
    # 同上
    val_data = MyDataset(istrain=0, transform=val_transforms)
    val_dataloader = DataLoader(val_data,batch_size=batch_size,shuffle=True,num_workers=4, prefetch_factor=8)

    _, train_loss_array, train_acc_array, val_loss_array, val_acc_array = training(model, "MyModel", num_epochs, train_dataloader, val_dataloader)


    visualize_training_results(train_loss_array, val_loss_array, train_acc_array, val_acc_array, num_epochs, "MyModel", batch_size)

