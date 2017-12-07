# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:41:38 2017

@author: guojm14
"""

import numpy as np
import tensorflow as tf 
from TPS_STN import TPS_STN
slim=tf.contrib.slim
def CNNpart(x,is_training=True):
    with tf.variable_scope("CNN"):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
            conv1=slim.conv2d(x, 64, 3, 1)
            pool1=slim.max_pool2d(conv1, [2, 2])
            print  pool1.shape
            conv2=slim.conv2d(pool1, 128, 3, 1)
            pool2=slim.max_pool2d(conv2, [2, 2])
            print pool2.shape
            conv3=slim.conv2d(pool2, 256, 3, 1)
            conv4=slim.conv2d(conv3, 256, 3, 1)
            pool3=slim.max_pool2d(conv4, [2, 2],[2,1],padding='SAME')
            print pool3.shape
            conv5=slim.conv2d(pool3, 512, 3, 1)
            conv6=slim.conv2d(conv5, 512, 3, 1)
            pool4=slim.max_pool2d(conv6, [2, 2],[2,1],padding='SAME')
            print pool4.shape
            conv7=slim.conv2d(pool4, 512, 2, 1,padding="VALID")
        
    return conv7
def Blstmpart(x,nh=10):
    with tf.variable_scope("BLSTM"):
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
    top=np.concatenate([np.expand_dims(np.linspace(-1,1,nx),1),np.full([nx,1],1)],1)
    bot=np.concatenate([np.expand_dims(np.linspace(-1,1,nx),1),np.full([nx,1],-1)],1)
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
    x=Blstmpart(x)
    print x
    a= attentionpart(x,hidden_size=4)
    print a
testcode()

        
    


