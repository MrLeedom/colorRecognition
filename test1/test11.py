import numpy as np
import cv2


def color_matrix(img):
        c = cv2.resize(img, (819, 460))
        b = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        thresh = (thresh / 255).astype(np.int8)
        thresh = 1-thresh
        return thresh


image = cv2.imread('F:\\maomi\\5.png')
ret = color_matrix(image)
print(ret[634,109])
cv2.imshow("images", ret)
#       cv2.imshow("haha",output)
cv2.waitKey(0)
