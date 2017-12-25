# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:41:38 2017

@author: guojm14
"""

import numpy as np
import tensorflow as tf 
from TPS_STN import TPS_STN
slim=tf.contrib.slim
from loaddata import dataloader
from utils import *
import os
# attention cell defined but not used
def CNNpart(x,is_training=True):
    with tf.variable_scope("CNN"):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
            conv1=slim.conv2d(x, 64, 3, 1)
            pool1=slim.max_pool2d(conv1, [2, 2])
            conv2=slim.conv2d(pool1, 128, 3, 1)
            pool2=slim.max_pool2d(conv2, [2, 2])
            conv3=slim.conv2d(pool2, 256, 3, 1)
            conv4=slim.conv2d(conv3, 256, 3, 1)
            pool3=slim.max_pool2d(conv4, [2, 2],[2,1],padding='SAME')
            conv5=slim.conv2d(pool3, 512, 3, 1)
            conv6=slim.conv2d(conv5, 512, 3, 1)
            pool4=slim.max_pool2d(conv6, [2, 2],[2,1],padding='SAME')
            conv7=slim.conv2d(pool4, 512, 2, 1,padding="VALID")
        
    return conv7
def Blstmpart(x,nh=256,index="0"):
    with tf.variable_scope("BLSTM"+index):
        x=tf.squeeze(x)
        T=x.shape[1]
        with tf.variable_scope('blstm1'):
            fw1 = tf.contrib.rnn.LSTMCell(nh)
            bw1 = tf.contrib.rnn.LSTMCell(nh)
            bi_lstm1 = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw1, cell_bw = bw1, inputs = x,dtype=tf.float32)
            #print bi_lstm1[0]
        
            temp1=tf.concat(bi_lstm1[0],2)
            #print temp1.shape
            temp1=tf.reshape(temp1,[-1,nh*2])
            #print temp1.shape
            out1=tf.reshape(slim.fully_connected(temp1, nh, activation_fn=None),[-1,T,nh])
            #print out1.shape
        with tf.variable_scope('blstm2'):
            fw2 = tf.contrib.rnn.LSTMCell(nh)
            bw2 = tf.contrib.rnn.LSTMCell(nh)
            bi_lstm2 = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw2, cell_bw = bw2, inputs = out1,dtype=tf.float32)
            temp2=tf.concat(bi_lstm2[0],2)
            temp2=tf.reshape(temp2,[-1,nh*2])
            out2=tf.reshape(slim.fully_connected(temp2, nh, activation_fn=None),[-1,T,nh])
    return out2

def local(x,is_training=True):
    with  tf.variable_scope('local'):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
            conv1=slim.conv2d(x, 64, 3, 1)
            pool1=slim.max_pool2d(conv1, [2, 2])
            conv2=slim.conv2d(pool1, 128, 3, 1)
            pool2=slim.max_pool2d(conv2, [2, 2])
            conv3=slim.conv2d(pool2, 256, 3, 1)
            pool3=slim.max_pool2d(conv3, [2, 2])
            conv4=slim.conv2d(pool3, 512, 3, 1)
            pool4=slim.max_pool2d(conv4, [2, 2])
            temp=slim.flatten(pool4, scope='flatten')
            fc1=slim.fully_connected(inputs=temp, num_outputs=1024, scope='fc1')
    return fc1
    
def fcforpoint(input_, nx=10,ny=2):
    
    shape = input_.get_shape().as_list()
    top=np.concatenate([np.expand_dims(np.linspace(-1,1,nx),1),np.full([nx,1],2)],1)
    top[0,0]=-2
    top[-1,0]=2
    bot=np.concatenate([np.expand_dims(np.linspace(-1,1,nx),1),np.full([nx,1],-2)],1)
    bot[0,0]=-2
    bot[-1,0]=2
    v=np.concatenate([bot,top],0)
    v=v.reshape(nx*ny*2)
    
    with tf.variable_scope("pointLinear"):
        matrix = tf.get_variable("Matrix", [shape[1], nx*ny*2], tf.float32,
                 tf.constant_initializer(0.0))
        bias = tf.get_variable("bias", [nx*ny*2],
      initializer=tf.constant_initializer(v))
    
    
    return tf.tanh(tf.reshape((tf.matmul(input_, matrix) + bias),[-1,nx*ny,2]))
def attentioncell(prev_hidden,x,cur_embeddings,is_training=True,reuse=True,hidden_size=256):
    with tf.variable_scope("attentioncell",reuse=reuse):
        B,T,H=x.shape
        xtemp=tf.reshape(x,[-1,H])
        xtemp1=slim.fully_connected(inputs=xtemp, num_outputs=hidden_size, scope='fc1')
        print xtemp1
        print prev_hidden
        prevtemp=slim.fully_connected(inputs=prev_hidden, num_outputs=hidden_size, scope='fc2')
        print prevtemp
        prevtemp1=tf.reshape(tf.tile(tf.reshape(prevtemp,[B,1,hidden_size]),[1,T,1]),[-1,hidden_size])
        emition=tf.reshape(slim.fully_connected(tf.tanh(prevtemp1+xtemp1),1,scope='score'),[B,T])
        alpha=tf.nn.softmax(emition)
        context=tf.reduce_sum((x*tf.tile(tf.reshape(alpha,[B,T,1]),[1,1,H])),axis=1)
        context=tf.concat([context,cur_embeddings],1)
        print context
        lstm_cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
        _,hidden=lstm_cell(context,prev_hidden)
    return hidden,alpha
def attentionpart(x,label=None,text=None,text_length=None,num_step=20,num_class=37,hidden_size=256,is_training=False):
    with tf.variable_scope("attention"):
        B,T,H=x.shape
        hidden = tf.zeros([B,hidden_size])
        out=tf.zeros([B,num_step,num_class])
        out=[]
        if not is_training:
            cur_embeddings=tf.zeros([B,num_class])
            for i in xrange(num_step):
                if i==0:
                    hidden,alpha=attentioncell(hidden,x,cur_embeddings,is_training=is_training,hidden_size=hidden_size,reuse=False)
                else:
                    hidden,alpha=attentioncell(hidden,x,cur_embeddings,is_training=is_training,hidden_size=hidden_size)
                cur_embeddings=tf.nn.softmax(slim.fully_connected(hidden,num_class))
                out.append(cur_embeddings)
        out=tf.reshape(tf.stack(out),[B,num_step,num_class])
    return out

def testcode():
    x=tf.ones([2,32,100,3])
    point=fcforpoint(local(x))
    print point
    print x
    x=TPS_STN(x,10,2,point,[32,100,3])
    print x
    x=CNNpart(x)
    print x
    x=Blstmpart(x,index='0')
    x=Blstmpart(x,index='1')
    x=tf.reshape(x,[-1,256])
    x=slim.fully_connected(x,37)
    x=tf.reshape(x,[24,2,37])
    print x


class CRNN(object):
    def __init__(self,sess,batch_size=16,
                num_epoch =200,
                lr=0.001,
                imagesize=[32,100],
                pointnum=[5,2],
                datapath='/home/guojm14/Downloads/IIIT5K/',
                trainlist='/home/guojm14/Downloads/IIIT5K/traindata.mat',
                testlist='/home/guojm14/Downloads/IIIT5K/testdata.mat'):
        self.sess=sess
        self.lr=lr
        self.batch_size=batch_size
        self.num_epoch=num_epoch
        self.pointnum=[5,2]
        self.imagesize=imagesize
        self.model_dir='version1'
        self.trainloader=dataloader(datapath,trainlist,batchsize=self.batch_size,t_name='train')
        self.testloader=dataloader(datapath,testlist,batchsize=self.batch_size,t_name='test',mode='testdata')
        self.trainloader.start()
        self.testloader.start()
        self.build_model() 
        self.saver = tf.train.Saver()
    def build_model(self):
        self.img=tf.placeholder(tf.float32,[self.batch_size,self.imagesize[0],self.imagesize[1],3])
        self.label= tf.sparse_placeholder(tf.int32)
        self.seq_len = tf.placeholder(tf.int32, [None])
        with tf.variable_scope("transform"):
            self.point=fcforpoint(local(self.img),nx=self.pointnum[0],ny=self.pointnum[1])
            #print self.point
            self.img_re=TPS_STN(self.img,self.pointnum[0],self.pointnum[1],self.point,self.imagesize+[3])
        with tf.variable_scope('reco'):
            x=CNNpart(self.img_re)
            x=Blstmpart(x,index='0')
            x=Blstmpart(x,index='1')
            x=tf.reshape(x,[-1,256])
            x=slim.fully_connected(x,37)
            self.labelp=tf.transpose(tf.reshape(x,[self.batch_size,24,37]),(1,0,2))
        
            self.loss = tf.nn.ctc_loss(labels=self.label,inputs=self.labelp, sequence_length=self.seq_len)
            self.cost=tf.reduce_mean(self.loss) 
            self.loss_sum=tf.summary.scalar('loss',self.cost)
        t_vars = tf.trainable_variables()
        self.l_vars = [var for var in t_vars if 'transform' in var.name]
        self.r_vars = [var for var in t_vars if 'reco' in var.name]
    def train(self):
        optimi=tf.train.AdamOptimizer(self.lr).minimize(self.loss,var_list=self.r_vars)
        optimil=tf.train.AdamOptimizer(self.lr*0.05).minimize(self.loss,var_list=self.l_vars)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        self.loss_sum1=tf.summary.merge([self.loss_sum])
        self.writer=tf.summary.FileWriter('./log',self.sess.graph)
        

        
        for i in xrange(int(self.trainloader.length/self.batch_size*self.num_epoch)):
            seq_len = np.ones(self.batch_size) * 24
            traindata,trainlabel=self.trainloader.getdata()
            if i%100==0:
                _,_,loss_str,loss=self.sess.run([optimi,optimil,self.loss_sum1,self.cost],feed_dict={self.img:traindata,self.label:trainlabel,self.seq_len:seq_len})
                self.writer.add_summary(loss_str,i)
                
                print 'epoch '+str(self.trainloader.epoch)+' iter '+str(i)+' loss '+str(loss)
            else:
                _,_=self.sess.run([optimi,optimil],feed_dict={self.img:traindata,self.label:trainlabel,self.seq_len:seq_len})
            if i%1000==0:
                testdata,testlabel=self.testloader.getdata()
                
                loss,reimg,point=self.sess.run([self.cost,self.img_re,self.point],feed_dict={self.img:testdata,self.label:testlabel,self.seq_len:seq_len})
                print 'test '+str(i)+'loss '+str(loss)
                print reimg.shape
                print point[0]
                save_images(reimg*255,[4,4],'sample/'+str(i)+'reimg.jpg')
                save_images(testdata*255,[4,4],'sample/'+str(i)+'img.jpg')   
                self.save('model/', i)
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
global_step=step)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
with tf.Session(config=run_config) as sess:
    a=CRNN(sess)

    a.train()        
    


