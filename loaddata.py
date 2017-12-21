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
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
 
    return indices, values, shape
def string2list(string):
    li=[]
    for i in xrange(len(string)):
        if string[i].isalpha():
            li.append(ord(string[i])-55)
        else:
            li.append(int(string[i]))
    return li
class dataloader(threading.Thread):
    def __init__(self,datapath,datalistfile,size=[100,32],batchsize=3,t_name='dataloader',mode='traindata'):
        threading.Thread.__init__(self, name=t_name)  
        self.datapath=datapath
        self.datalist= io.loadmat(datalistfile)[mode][0]
        self.dataqueue=Queue.Queue(maxsize=10)
        self.bs=batchsize
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
            
            label=sparse_tuple_from(map(string2list,label))
            data=np.array(data)
            item=(data,label)
            self.dataqueue.put(item)
    def getdata(self):
        return self.dataqueue.get()
    def close(self):
        self.on=False
def testcode():
    a=dataloader('/home/guojm14/Downloads/IIIT5K/','/home/guojm14/Downloads/IIIT5K/testdata.mat',mode='testdata')
    a.start()
    data,label= a.getdata()
    print data.shape
    print label
    a.close()
testcode()
