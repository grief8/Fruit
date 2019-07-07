import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms


# 数据归一化
def data_in_one(inputdata):
    inputdata = (inputdata - inputdata.min()) / (inputdata.max() - inputdata.min())
    return inputdata


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, fname, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        data = []
        file = open(os.path.join(root, fname))
        for line in file:
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            # eeg = np.loadtxt(os.path.join(root, words[0]), delimiter=',')
            data.append((words[0], int(words[1])))
        self.root = root
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.data[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # eeg = torch.from_numpy(np.loadtxt(os.path.join(self.root, fn), delimiter=','))
        img = Image.open(os.path.join(self.root, fn)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            # eeg = eeg.type(torch.FloatTensor)  # 转Float
            img = img.cuda()  # 转cuda
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)


class ClassDataset:
    def __init__(self, path, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        data = []
        file = open(path)
        for line in file:
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            # eeg = np.loadtxt(os.path.join(root, words[0]), delimiter=',')
            data.append((words[0], int(words[1])))
        self.path = path
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.data[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # eeg = torch.from_numpy(np.loadtxt(os.path.join(self.root, fn), delimiter=','))
        img = Image.open(fn.encode('gbk')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            # eeg = eeg.type(torch.FloatTensor)  # 转Float
            img = img.cuda()  # 转cuda
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)
