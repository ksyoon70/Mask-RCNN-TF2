# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:30:16 2021

@author: headway
"""
import numpy as np
import tensorflow as tf
import keras.backend as K

rpn_match = tf.constant([[[1], [0],[0]],[[1],[0],[-1]]])

rpn_bbox =tf.constant([[[0.1,0.1,0,0],[0,0,0,0],[0,0,0,0]],[[0.2,0.2,0,0],[0,0,0,0],[0,0,0,0]]])


target_bbox=tf.constant([[[0.2,0.2,0,0],[0,0,0,0],[0,0,0,0]],[[0.3,0.3,0,0],[0,0,0,0],[0,0,0,0]]])

rpn_match = K.squeeze(rpn_match, -1)

indices = tf.where(K.equal(rpn_match, 1))

rpn_bbox = tf.gather_nd(rpn_bbox, indices)

batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)

def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

target_bbox = batch_pack_graph(target_bbox, batch_counts,2)