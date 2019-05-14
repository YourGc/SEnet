# coding:utf-8
from model import se_resnet_50
import torch
from Data_Loader import dataloader
if __name__ == '__main__':
    model = torch.load('epoch_82.pth')
    model.train(False)
    for tx,ty in dataloader['val']:
        output = model(tx)
        print(output.data)