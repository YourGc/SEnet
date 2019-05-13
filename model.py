# coding:utf-8

import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    #resnext 32x4d
    def __init__(self):
        super(Model,self).__init__()
        self.model = build_model()

    def forward(self, x):
        pass

