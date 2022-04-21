import torch
import collections
import numpy as np
from pathlib import Path
import os
from lxml import etree
# a = np.array([1, 2, 3])
# print(a, type(a))
# b = torch.from_numpy(a)
# print(b, type(b))
# print(torch.cuda.is_available())
# print(4 // 3)
# a = ['b', 'c', 'a']
# b = [i for i in a if i and not 'b']
# for i in b:
#     print(i)
# with open('C:/Users/Mr.Feng/Desktop/test.txt', "r") as f:
#     lines = f.read().split('\n')  # 通过换行符进行分割，变成一行一行的数据
# print(lines)
# lines = [x for x in lines if x and not x.startswith("#")]
# print(lines)
# for line in lines:
#     print(line)
# a = [1, 2, 3, 4, 5, 6, 7, 8]
# b = [float(x) for x in a]
# print(b)
# with open('D:/VOCdevkit/VOC2012/Annotations/2008_006188.xml', 'r') as f:
#     xml_str = f.read()
# print(xml_str)
# xml = etree.fromstring(xml_str)
# print(xml)
# for i in xml:
#     print(i)
#     print(len(i))
#     print(i.tag, i.text)
# print(os.sep)
# print(3 // 2)
# a = []
# a.append({})
# a[-1]['a'] = 1
# print(a)
# print((2, )*3)
# a = 0
# for i in range(5):
#     a += i
# print(a)
# a = [1, 2, 3]
# print(a[-1], a[-2])
a = torch.randn(2, 3)
print(a)
b = a.view(3, 2)
print(b)
print(b.view(2, 3))
a = collections.OrderedDict({'a': 2, 'b': 3, 'c': 4})
print(a['a'])
a = torch.tensor(0.5)
print(a, a.view(1, 1))
c = a.view(1, 1)
b = torch.tensor([[1, 1], [1, 1]])
print(b)
print(b*c)
a = np.floor(np.arange(10) / 4).astype(int)
print(a, type(a))
print(np.floor(1.1))
# a = torch.randn(2, 3, 2, 2)
# print(a)
# print(a.stride())
# print(a.storage())
a = np.array([1, 2, 4, 3])
print(a*2)
print(np.ceil(1.5))
a = Path('./my_yolo_dataset/train/labels/2009_004012.txt').parent
print(a, type(a))
print(os.path.isdir('./my_yolo_dataset/train/labels'))
print(os.getcwd())
print(os.path.isabs(a))
a = np.load('./my_yolo_dataset/train/labels.norect.npy', allow_pickle=True)
l = a[0]
print(l, l.shape)
print(l>=0)
b = np.array([[1, 2, 3], [1, 1, 2]])
print(b.shape)
print(np.unique(l, axis=1))
print(int(1.8))
print(np.eye(3))
a = np.eye(3)
b = a
b[:, 2] = 2
print(b)
print(a)
i = (2>1)&(3>2)
print(i)
print(a[i])
a = np.arange(6)
a.shape = (2, 3)
print(a)
b = a>1
print(b)
print(a, a[b])
print(5717 % 4)
with open('D:/VOCdevkit/VOC2012/ImageSets/Main/val.txt', 'r') as f:
    a = f.read().splitlines()
print(len(a))
a = torch.tensor(123.8)
print(a, a.long())
a = np.random.randn(3, 2)
print(a)
print(a.T)
print(max(round(64 / 4), 1))
a = torch.tensor([0])
print(a)
b= torch.tensor([1])
print(torch.cat((a, b)))
a = torch.randn((1, 2, 2))
print(a, a[:, 1])
a = 123.2
print(round(a), type(round(a)))