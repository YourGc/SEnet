# coding:utf-8
#from model import se_resnet_50
import torch
import torch.nn.functional as F
from Data_Loader import dataloader
if __name__ == '__main__':
    model = torch.load('./output/epoch_82.pth')
    model.train(False)
    error_sum  =0
    for index,(tx,ty) in enumerate(dataloader['val']):
        ty = torch.autograd.Variable(ty).cuda()
        output = model(tx)
        output = F.softmax(output)
        _,pred = torch.max(output,1)
        error_sum += torch.sum(pred !=ty)

    print(error_sum/len(dataloader['val']))
