import numpy as np
import tensorflow as tf 
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

def local(x):
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
def fcforpoint()
    
def fcforpoint(input_, output_size, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope("pointLinear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
return tf.matmul(input_, matrix) + bias
def testcode():
    x=tf.ones([2,32,100,3])
    x=CNNpart(x)
    print x
    x=Blstmpart(x)
    print x
testcode()

        
    


