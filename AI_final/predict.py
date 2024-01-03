import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt

from get_data import MyDataset
import pandas as pd
# from  models.ResNet import resnet50
# from models.EfficientNet import efficientnet_b0
import numpy as np
from models.ResNet import Multimodal
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
# import albumentations
# import torchvision.models as models

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

# 生成混淆矩阵（由预测结果）
    def update(self, preds, labels):
        preds = torch.argmax(preds, 1)
        preds.to('cpu').numpy()
        labels.to('cpu').numpy()
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
            #self.martrix[t, p] += 1

    def summary(self):
        # calculate accuracy
        p=[0,0,0]
        r=[0,0,0]
        f1=[0,0,0]
        sum_TP = 0
        #self.matrix= self.matrix.numpy()
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("Our model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            # round(数值，保留的小数位数）
            #Accuracy = round((TP+TN)/(TP+FP+FN+TN),3) if TP+FP+FN+TN !=0 else 0.
            # 修改位数
            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            F1 = round((2*Precision*Recall) / (Precision + Recall), 4) if Precision+Recall != 0 else 0.
             # 为了计算 平均值
            p[i] = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            r[i] = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            f1[i] = round((2 * Precision * Recall) / (Precision + Recall), 4) if Precision + Recall != 0 else 0.
            #Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i],  Precision, Recall, F1]) # 添加行
        # 最后添加一行
        table.add_row(["macro_average", np.average (p), np.average (r), np.average (f1)])  # 添加行
        print(table)

    def plot(self):
        plt.clf()
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        # 新增代码
        proportion = []
        #proportion = np.zeros((self.num_classes, self.num_classes))
        matrix=matrix.T

        for i in matrix:
           for j in i:
             temp = j / (np.sum(i))
             proportion.append(temp)

        pshow = []
        for i in proportion:
            pt = "%.2f%%" % (i * 100)
            pshow.append(pt)
        #注意转置的使用
        proportion = np.array(proportion).reshape(self.num_classes, self.num_classes).T  # reshape(列的长度，行的长度)
        pshow = np.array(pshow).reshape(self.num_classes, self.num_classes).T

        plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        iters = np.reshape([[[i, j] for j in range(self.num_classes)] for i in range(self.num_classes)], (matrix.size, 2))
        for i, j in iters:
            if (i == j):
                plt.text(j, i - 0.12, int(matrix[j, i]), va='center', ha='center', fontsize=12,
                         color='white', weight=5)  # 显示对应的数字
                plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='white')
            else:
                plt.text(j, i - 0.12, int(matrix[j, i]), va='center', ha='center', fontsize=12)  # 显示对应的数字
                plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

        plt.savefig("./MyModel_ConfusionMatrix.png")

        for x in range(self.num_classes):

           for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
              info = int(matrix[y, x])
              plt.text(x, y, info,
                        verticalalignment='center', #垂直居中
                        horizontalalignment='center', #水平居中
                        color="white" if info > thresh else "black")
        #plt.tight_layout()
        #plt.show()


def main():
    batch_size = 16

    prelabel_list = []
    likelihood_list = []
    label_list = []
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")


    model = Multimodal()
    # model = efficientnet_b0(num_classes=2)
    # model = resnet50(num_classes=2)
    model.to(device)
    best_path = './results/trained_model/MyModel.pth'
    weights = torch.load(best_path)
    
    model.load_state_dict(weights, strict=False)

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]


        # Testing dataset for the current fold
    test_dataset = MyDataset(istrain=2, transform=test_transforms)
    print(len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    model.eval()
    testlabelst = []
    prob_list = []
    with torch.no_grad():
        for x,f1, f2, y in test_dataloader:
            x,f1,f2, y= x.to(device),f1.to(device), f2.to(device), y.to(device)
            testlabelst.append(y)
            pred = model(x)
            outputs = pred

            confusion.update(outputs, y)
        
            predict = torch.softmax(pred, 1).cpu()
            predict_cla = torch.argmax(predict, 1)

            label_list.append(y)
            prelabel_list.append(predict_cla)
            likelihood = torch.max(predict, dim=1)
            likelihood_list.append(likelihood)
            prob_list.append(predict[:,1])

    label_array = np.concatenate([tensor.cpu().numpy() for tensor in label_list])
    likelihood_array = np.concatenate([tensor.cpu().numpy() for tensor in prob_list])

    auc = roc_auc_score(label_array, likelihood_array)
    print("AUC: ", auc)

    confusion.plot()
    confusion.summary()    




if __name__ == '__main__':
    plt.switch_backend('agg')
    main()

