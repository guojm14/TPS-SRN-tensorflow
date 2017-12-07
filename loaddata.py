# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:41:38 2017

@author: guojm14
"""

from  scipy import io
import os
import threading
import Queue
import numpy as np
from PIL import Image
class dataloader(threading.Thread):
    def __init__(self,datapath,datalistfile,size=[100,32],batchsize=64,t_name='dataloader'):
        threading.Thread.__init__(self, name=t_name)  
        self.datapath=datapath
        self.datalist= io.loadmat(datalistfile)['traindata'][0]
        self.dataqueue=Queue.Queue(maxsize=10)
        self.bs=64
        self.on=True
        self.index=0
        self.length=len(self.datalist)
        self.epoch=0
        self.size=size
        print 'inited'
    def run(self):
        while(self.on):
            
            data=[]
            label=[]
            for i in xrange(self.bs):
                imgname=str(self.datalist[self.index][0][0])
                img=np.array(Image.open(os.path.join(self.datapath,imgname)).resize(self.size))
                imlabel=str(self.datalist[self.index][1][0])
                data.append(img)
                label.append(imlabel)
                self.index+=1
                if self.index==self.length:
                    self.index=0
                    self.epoch+=1
            label=np.array(label)
            data=np.array(data)
            item=(data,label)
            self.dataqueue.put(item)
    def getdata(self):
        return self.dataqueue.get()
    def close(self):
        self.on=False
def testcode():
    a=dataloader('','traindata.mat')
    a.start()
    data,label= a.getdata()
    print data.shape
    a.close()
#testcode()
