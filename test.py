import tensorflow as tf
import numpy as np
from PIL import Image
from TPS_STN import TPS_STN

img = np.array(Image.open("6_7.png"))
out_size = list(img.shape)
shape = [1]+out_size

nx=5
ny=2
top=np.concatenate([np.expand_dims(np.linspace(-1,1,5),1),np.full([5,1],1)],1)
bot=np.concatenate([np.expand_dims(np.linspace(-1,1,5),1),np.full([5,1],-1)],1)
v = np.array([
  [0.2, 0.2],
  [0.4, 0.4],
  [0.6, 0.6],
  [0.8, 0.8]])
v=np.concatenate([bot,top],0)
print v
p = tf.constant(np.concatenate([v.reshape([1, nx*ny, 2]),v.reshape([1, nx*ny, 2])],0) ,dtype=tf.float32)
p=tf.constant(v.reshape([1, nx*ny, 2]),dtype=tf.float32)
t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
print out_size
print p
t_img = TPS_STN(t_img, nx, ny, p, [35,93,3])
print t_img.shape
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  img1 = sess.run(t_img)
  #print img1
  Image.fromarray(np.uint8(img1.reshape(out_size))).save("transformed.png") 
