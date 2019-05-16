# coding:utf-8
import pickle
import os
import json
from PIL import Image
from Data_Loader import preprocess
from progressbar import *
class Cache():
    def __init__(self,file,json_path = '.',test_dir = 'Test_fix'):
        self.file = file
        if not os.path.exists(file):
            self.cache = self.make_cache(file,json_path,test_dir)

        else :self.cache = pickle.load(file)

    def make_cache(self,file,json_path,test_dir):
        roi = {}
        roi_anno = self.load_json(json_path)
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets,maxval=len(roi_anno)).start()
        for index, (k, v) in enumerate(roi_anno.items()):
            pbar.update(index)
            img_ori = Image.open(os.path.join(test_dir, k))
            tmp = []
            for coords in v:
                img = img_ori.crop(coords).resize((64, 64))
                img = preprocess(img)
                img = img.unsqueeze(0)
                tmp.append(img)
            roi[k] = tmp
        with open(file,'r') as f:
            pickle.dump(roi,f)
        f.close()
        return roi
    def load_json(self,path = '.'):
        with open(os.path.join(path, 'roi.json'), 'r')as f:
            rois = json.load(f)
        f.close()
        return rois