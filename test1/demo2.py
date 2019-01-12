# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:46:24 2018

@author: leedom
"""

import cv2
impath="F:\\maomi\\aim.png"
img=cv2.imread(impath)

print(img.shape)
cv2.rectangle(img,(100,100),(400,400),(0,255,0),3)
cv2.imwrite("F:\\maomi_resize\\aim.png", img)
cv2.imshow("Image", img)   

cv2.waitKey (0)  

cv2.destroyAllWindows() 