# coding:utf-8
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))##VOC dataset 224*224
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

class dataset(Dataset):
    def __init__(self, path = 'SE_train' ,loader=default_loader):
        #定义好 image 的路径
        self.images,self.labels = get_img(path)
        self.loader = loader

    def __getitem__(self, index):
        fp = self.images[index]
        img = self.loader(fp)
        label = self.labels[index]
        return img,label

    def __len__(self):
        return len(self.images)

train_data = dataset('SE_train')
trainloader = DataLoader(train_data, batch_size=4, shuffle=True)

valdata = dataset('SE_val')
valloader = DataLoader(valdata, batch_size=4, shuffle=True)

dataloader = {
    "train":trainloader,
    "val":valloader
}