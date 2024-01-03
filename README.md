# 生物医学大数据的人工智能基础课程大作业
## 基于卷积神经网络的胸部X射线结核病诊断

get_data.py：获取数据集

train_singlemode.py：模型训练代码，单模态（只使用图像数据）

train_multi.py: 模型训练代码，多模态（使用图像+文本） *旋转增强也在train_ .py里的 transforms 中实现*

predict.py：测试代码。得到Acc，AUC混淆矩阵

augmentation.py：使用CLAHE数据增强。

text_pre.py：对文本数据预处理：提取年龄和性别数据并进行one-hot编码，保存到了encoded_features.csv中。


models目录：

models.ResNet.py：ResNet模型代码，*多模态模型的代码也在这个文件的最后 (class Multimodal(nn.Module)）*。

models.EfficientNet.py：EfficientNet模型代码。
