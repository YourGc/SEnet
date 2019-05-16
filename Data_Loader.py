# coding:utf-8
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import json
import os
import numpy as np
import pandas as pd

mean = [0.22116, 0.2263, 0.23685]
std = [0.25391, 0.27520, 0.30595]
normalize = transforms.Normalize(
    mean= mean,
    std= std
)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((64,64)) #change to 64
    img_tensor = preprocess(img_pil)
    return img_tensor

    
def get_img(path):
    labels_dirs = os.listdir(path)
    imgs_name =[]
    labels = []
    for i in labels_dirs:
        cur_path = os.path.join(path,str(i))
        i_imgs_names = os.listdir(cur_path)
        imgs_name += [os.path.join(cur_path,img_name) for img_name in i_imgs_names]
        labels += [i] * len(i_imgs_names)

    random_seed = random.randint(0, 100)
    random.seed(random_seed)
    random.shuffle(imgs_name)
    random.seed(random_seed)
    random.shuffle(labels)

    return imgs_name,labels

def load_json(self,path):
    with open(os.path.join(path, 'roi.json'), 'r')as f:
        rois = json.load(f)
    f.close()
    return rois

class dataset(Dataset):
    def __init__(self,csv_path,dir):
        #定义好 image 的路径
        self.annos = self.load_csv(csv_path)
        self.imgs = os.listdir(dir)

    def load_csv(self,path):
        annos = pd.read_csv(path)
        return annos

    def train_loader(self,fn):
        anno = self.annos[self.annos['filename'] == fn]
        x1,y1,x2,y2 = anno['X1'],anno['Y1'],anno['X2'],anno['X2']
        img_pil = Image.open(fn)
        img_croped = img_pil.crop((x1,y1,x2,y2))
        img_tensor = preprocess(img_croped)
        return img_tensor,int(anno['type'])

    def __getitem__(self, index):
        fn = self.imgs[index]
        img,label = self.train_loader(fn)
        return img,label

    def __len__(self):
        return len(self.imgs)

# class dataset(Dataset):
#     def __init__(self, path = 'SE_train' ,loader=default_loader):
#         #定义好 image 的路径
#         self.images,self.labels = get_img(path)
#         self.loader = loader
#
#     def __getitem__(self, index):
#         fp = self.images[index]
#         img = self.loader(fp)
#         label = self.labels[index]
#         return img,label
#
#     def __len__(self):
#         return len(self.images)

# Batch_size = 4
# train_data = dataset(csv_path='train_label_fix.csv',dir = 'Train_fix')
# trainloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True)
#
# valdata = dataset(csv_path='val_label_fix.csv',dir = 'Val_fix')
# valloader = DataLoader(valdata, batch_size=Batch_size, shuffle=True)
#
#
# dataloader = {
#     "train":trainloader,
#     "val":valloader
# }