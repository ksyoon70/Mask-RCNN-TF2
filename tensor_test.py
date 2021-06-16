# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:16:56 2021

@author: user
"""
import numpy as np

rois = np.random.randint(4,size=(1,2,4))
print(rois)

image_meta =  np.random.randint(5,size=(2,14))
print(image_meta)

result1 = [rois,image_meta]
#print(result1)

freaturemaps = [ "P1" , "P2", "P3"]

result2= result1 + freaturemaps

print(result2)