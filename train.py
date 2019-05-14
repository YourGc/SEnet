# coding:utf-8

from Data_Loader import dataloader
import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from model import se_resnet_50

def train(model,opt,loss,lr_decay,epochs):

    for epoch in range(epochs):

        for i,data in enumerate(dataloader['train']):

if __name__ == '__main__':

    #gpu availabel
    is_GPU = torch.cuda.is_available()

    #modules
    model = se_resnet_50()
    model.train(True)
    loss = nn.CrossEntropyLoss()
    opt = opt.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.00003)
    lr_decay = torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=0.95)
    train(model,opt,loss,lr_decay,epochs=200)
