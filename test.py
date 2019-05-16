
# coding:utf-8
import json
import torch.nn as nn
from PIL import Image
from Data_Loader import preprocess
import torch
import torch.nn.functional as F
import pandas as pd
from progressbar import *
from cache import Cache
from torchsummary import summary
from se_resnext import se_resnext_50
from se_resnet import se_resnet_50
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
    #rois = load_json('.')
    # model = se_resnext_50()
    #torch.save(model,'test.pkl')
    cache = Cache('./Cache/roi.pkl','roi.json',r'F:\小目标检测\Test_fix')
    model = torch.load('epoch_9.pth.tar')
    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    # print(model)
    # summary(model,(3,128,128))
    model.eval()
    model.cuda()
#    summary(model,(3,224,224))
    result = []
    error = []
    test_dir = '/home/star/Wayne/transportation/data/Test_fix'
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    # pbar = ProgressBar(widgets=widgets, maxval=len(rois)).start()
    # with torch.no_grad():
    #     for index,(k,v) in enumerate(rois.items()):
    #         pbar.update(index)
    #         img_ori = Image.open(os.path.join(test_dir,k))
    #         tmp = []
    #         for coords in v:
    #             img = img_ori.crop(coords).resize((64,64))
    #             img = preprocess(img)
    #             img = img.unsqueeze(0).cuda()
    # #            print(img)
    #             output = model(img)
    #
    #             output = F.softmax(output)
    #             conf, pred = torch.max(output, 1)
    #
    #             tmp.append([conf.float(),pred.int(),coords])
    #
    #         if len(tmp) == 0 :
    #             error.append(k)
    #             continue
    #         tmp = sorted(tmp,key=lambda x:x[0],reverse=True)
    #         X1,Y1,X2,Y2 = tmp[0][2]
    #         result.append([k,X1,Y1,X2,Y1,X2,Y2,X1,Y2,tmp[0][1]])
    #
    # print(error)
    # result = pd.DataFrame(result,columns=['filename','X1','Y1','X2','Y2',\
    #                                       'X3','Y3','X4','Y4','type'])
    # result.to_csv('result.csv',index=False)





