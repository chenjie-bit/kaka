import config as CF
import model as m
import glob
import os
import time
import numpy as np

def load_data(file_):
    data=[]
    file_lst=[]
    for line in open(file_,"r",encoding="utf-8"):
        da = line.split('\t')
        # print(len(da[0].split(",")))
        da = da[0].strip(",").split(",")
        # print(len(da))
        # print(type(da))
        # exit()
        da_=[]
        for x in da:
            try:
                da_.append(float(x))
            except:
                print(x)
                da_.append(0.0)
        data.append(da_)
        # print(np.array(data).shape)
        # exit()
    for i in range(256):
        file_lst.append("sample_"+str(i))
    return file_lst,data

st=time.time()
file_lst,data=load_data("raw_data.txt")


print("sample number is %s, sample dim is %s"%(len(data),len(data[0])))
class_num=CF.config["class_num"]
for model_type in CF.config["type"]: 
    model= m.model(class_num=class_num)
    model.build(model_type)
    model.run(data)
    model.show(file_lst)

print("聚类完成，一共用时:%s秒"%(time.time()-st))
