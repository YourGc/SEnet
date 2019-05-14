# coding:utf-8
import json
with open('roi_20190512T0913.json','r') as f1:
    result1 = json.load(f1)
f1.close()

with open('roi_20190512T2000.json','r') as f2:
    result2 = json.load(f2)
f2.close()

for k,v in result1.items():
    result2[k] = v

with open('roi.json','w') as f:
    json.dump(result2,f,indent=4)
f.close()