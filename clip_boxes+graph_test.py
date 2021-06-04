# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:59:32 2021

@author: user
"""
import tensorflow as tf
import numpy as np


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

window = np.array([0, 0, 1, 1], dtype=np.float32)
#window = np.array([[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1]], dtype=np.float32)

#boxes = np.array([0.1,0.1,0.5,0.5]) 
boxes = np.array([[0.1,0.1,0.5,0.5],[0.2,0.2,0.6,0.6],[0.3,0.3,0.7,0.7]],dtype=np.float32)
#boxes = np.array([(0.1,0.1,0.5,0.5),(0.2,0.2,0.6,0.6),(0.3,0.3,0.7,0.7)])
boxes = clip_boxes_graph(boxes, window)

print(boxes)