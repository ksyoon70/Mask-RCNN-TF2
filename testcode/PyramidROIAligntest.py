# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:32:41 2021

@author: user
"""
import tensorflow as tf
import numpy as np

#공식 싸이트 샘플
BATCH_SIZE = 3
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
CROP_SIZE = (24, 24)

image = tf.random.normal(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
CHANNELS) )
boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
maxval=BATCH_SIZE, dtype=tf.int32)
#박스와 박스의 갯수 만큼의 인덱스
output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)



#입력 변수
roi_level = [5,3,4,2]  # 임시로 P3에 해당한다고 한다.
boxes = np.array([[0.1,0.1,0.5,0.5],[0.2,0.2,0.6,0.6],[0.3,0.3,0.7,0.7],[0.4,0.3,0.7,0.7]],dtype=np.float32)
pool_shape = (3,3)
batch = 4

P5 = np.arange(batch*2*2*3).reshape(batch,2,2,3)
P4 = np.arange(batch*14*14*3).reshape(batch,14,14,3)
P3 = np.arange(batch*28*28*3).reshape(batch,28,28,3)
P2 = np.arange(batch*56*56*3).reshape(batch,56,56,3)

feature_maps = [ P5,P4,P3,P2]

pooled = []
box_to_level = []
for i, level in enumerate(range(2, 6)):
 ix = tf.where(tf.equal(roi_level, level)) #조건이 맞으면 인덱스 리턴
 level_boxes = tf.gather_nd(boxes, ix)  #Gather slices from params into a Tensor with shape specified by indices. boxes에서 ix 인덱스에 해당 값만 가져온다.

 # Box indices for crop_and_resize.
 box_indices = tf.cast(ix[:, 0], tf.int32)

 # Keep track of which box is mapped to which level
 box_to_level.append(ix)


 # Crop and Resize
 # From Mask R-CNN paper: "We sample four regular locations, so
 # that we can evaluate either max or average pooling. In fact,
 # interpolating only a single value at each bin center (without
 # pooling) is nearly as effective."
 #
 # Here we use the simplified approach of a single value per bin,
 # which is how it's done in tf.crop_and_resize()
 # Result: [batch * num_boxes, pool_height, pool_width, channels]
 pooled.append(tf.image.crop_and_resize(
     feature_maps[i], level_boxes, box_indices, pool_shape,
     method="bilinear"))

 print(pooled[i].shape)