import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
import torch
import os
from tqdm import tqdm

#遍历 FakeNewsNet 数据集中的新闻图片，使用 ResNet 提取特征，并将特征保存为文本文件。


#防止加载损坏的图片时报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

#构建一个神经网络 Net，用于提取图片特征
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        #去掉 ResNet 最后的全连接层，只保留卷积层部分，提取特征。
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    #定义前向传播，输入 x 通过 resnet_layer 提取特征
    def forward(self, x):
        x = self.resnet_layer(x)
        return x

# resnet50 = models.resnet50(pretrained=True, progress=True)
#加载 ResNet-18 预训练模型（已经在 ImageNet 上训练）
resnet18 = models.resnet18(pretrained=True, progress=True)
#使用 Net 进行封装，使其成为特征提取器
model = Net(resnet18)
#打印模型结构，输出 ResNet-18 的特征提取部分
print(model)  # output size 16*512*1*1

# 定义自定义数据集 img_Dataset
class img_Dataset(Dataset):
    def __init__(self, root, resize):
        #获取目录 root 下所有图片路径（支持 .jpg, .png, .JPG）
        self.image_files = np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        #定义图片缩放操作，将图片调整为 resize × resize 大小
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])
        #将图片转换为张量（Tensor）
        self.toTensor = transforms.ToTensor()
    #读取 index 位置的图片并转换格式，异常情况下返回空白图片（3×224×224）
    def __getitem__(self, index):
        path = self.image_files[index]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            img = self.toTensor(img)
        except:
            img = np.zeros((3, 224, 224))
            img = torch.Tensor(img)
        return img
    #返回数据集中的图片数量
    def __len__(self):
        return len(self.image_files)

#读取新闻数据集
ds_path = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset'
datasets = ['politifact', 'gossipcop']
datasub=['real','fake']
#用于存储每个数据集的新闻 ID
all_stats = {k1 : {k2:[] for k2 in datasub} for k1 in datasets}

#遍历politifact 和 gossipcop 的 real 和 fake 目录，提取新闻ID
for ds in datasets:
    for dsub in datasub:
        news_list = os.listdir(os.path.join(ds_path, ds,dsub))
        for nidpath in news_list:
            if nidpath[:len("politifact")]=="politifact":
                nid=nidpath[len("politifact"):]
            else:
                nid=nidpath[len("gossipcop-"):]
            all_stats[ds][dsub].append(nid)
#图片存放路径
file_paths = ["FakeNewsNet-Dataset/NewsImages/gossipcop_images",
              "FakeNewsNet-Dataset/NewsImages/politifact_images"]
#提取出的特征保存路径
file_out="processed_data/FakeNewsNet/"
#遍历 file_paths，加载 gossipcop 和 politifact 的所有图片
for file_path in tqdm(file_paths,desc='file_path'):
    all_img = os.listdir(file_path)
    #使用 img_Dataset 读取图片数据，设置尺寸 224×224
    dataset = img_Dataset(file_path, 224)
    #DataLoader 加载数据，batch_size=1 表示一次处理一张图片
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    EPOCH = 1
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):
            #取出图片文件名（去掉后缀）
            txt_name = all_img[step][:-4]
            #送入 ResNet 提取特征
            out = model(data)
            ## 重新调整张量形状
            out = torch.reshape(out, (out.shape[1], -1))
            print(step)
            #转换为 NumPy 数组
            out_np = out.detach().numpy()
            #根据图片名称找到对应的新闻类别，将提取出的特征保存为 .txt 文件
            for key,value in all_stats.items():
                for key2,value2 in value.items():
                    if txt_name in value[key2]:
                        np.savetxt(file_out+key + '/visual_feature/' + txt_name + '.txt', out_np)
                        break
#创建一张全零图片，提取其特征并保存为 white.txt
for key, value in all_stats.items():
    empty = np.zeros((1,3,224,224))
    empty = torch.Tensor(empty)
    out = model(empty)
    out = torch.reshape(out, (out.shape[1], -1))
    out_np = out.detach().numpy()
    np.savetxt(file_out+key + '/visual_feature/' + 'white.txt', out_np)
