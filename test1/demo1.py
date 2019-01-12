# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:42:03 2018

@author: leedom
"""

from skimage import io,data,color
img=data.coffee()
img_gray=color.rgb2gray(img)
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j]<=0.5):
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1
io.imshow(img_gray)