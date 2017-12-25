from __future__ import division
import math
import json
import os
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
def save_images(images, size, image_path):
  # images = (images+1.)/2.
  puzzle = merge(images, size)
  return scipy.misc.imsave(image_path, puzzle)

def merge(images, size):
  cdim = images.shape[-1]
  h, w = images.shape[1], images.shape[2]
  if cdim == 1 :
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j*h:j*h+h, i*w:i*w+w] = np.squeeze(image)
    return img
  else:
    img = np.zeros((h * size[0], w * size[1], cdim))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img
