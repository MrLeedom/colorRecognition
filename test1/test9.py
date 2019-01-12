import cv2

img = cv2.imread('F:\\maomi\\4.png')
cv2.imshow('ori',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

AA,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)  
#第一个参数是指明在哪幅图像上绘制轮廓；
#第二个参数是轮廓本身，在Python中是一个list。
#第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓
print(len(contours[0]))#轮廓的个数
print(len(contours[1]))
print(hierarchy)
cv2.imshow("img", img)
cv2.waitKey(0)



#错误提示，诗鹏
#ValueError: too many values to unpack 类错误，多为输入或者输出参数数量不一致导致。
