# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:27:20 2021

@author: headway
"""

import numpy as np
import tensorflow as tf
# 레벨을 표시
roi_level = tf.constant([2,4,5,2,4,3,5])
box_to_level = []

for i, level in enumerate(range(2, 6)):
    ix = tf.where(tf.equal(roi_level, level))
    box_to_level.append(ix)

box_to_level = tf.concat(box_to_level, axis=0)
print("box_to_level 표시")
print(box_to_level)

box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
print("box_range")
print(box_range)
box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)
print("box_to_level")
print(box_to_level)
sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
print("sorting_tensor")
print(sorting_tensor)
ixx =tf.nn.top_k(sorting_tensor, k=tf.shape(  #k 갯수만큼 큰것 부터 소팅 한다.
            box_to_level)[0])
print("ixx")
print(ixx)
ix = tf.nn.top_k(sorting_tensor, k=tf.shape(  #k 갯수만큼 큰것 부터 소팅 한다.
            box_to_level)[0]).indices[::-1]

print("ix")
print(ix)

ix = tf.gather(box_to_level[:, 1], ix)


test = tf.constant([[[1,2,3],[4,5,6]]])