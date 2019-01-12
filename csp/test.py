# coding:UTF-8
import cv2  # cv2是opencv官方的一个扩展库，里面含有各种有用的函数以及进程，opencv是一个基于开源发行的跨平台计算机视觉库，它轻量级而且高效
import numpy as np  # Numeric Python，它是由一个多维数组对象和用于处理数组的例程集合组成的库。Numpy拥有线性代数和随机数生成的内置函数
from skimage import draw
import matplotlib.pyplot as plt

class Detect:
    def __init__(self, path):
        # 原始图像信息
        self.ori_img = cv2.imread(path)
        # self.ori_img = path
        self.gray = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2HSV)
        # 获得原始图像行列
        rows, cols = self.ori_img.shape[:2]

        # 工作图像
        self.work_img = cv2.resize(
            self.ori_img, (int(cols / 4)*5, int(rows / 4)*5))
        self.work_gray = cv2.resize(
            self.gray, (int(cols / 4)*5, int(rows / 4)*5))
        self.work_hsv = cv2.resize(
            self.hsv, (int(cols / 4)*5, int(rows / 4)*5))
    #方案一：用skimage包
    def obstacle(self):
        
        X = [100,100,190,190]
        Y = [100,190,190,100]
        rr,cc = draw.polygon(Y,X)
        draw.set_color(self.ori_img,[rr,cc],(255,0,0))
        plt.imshow(self.ori_img)
        cv2.imwrite('./obstacle.jpg',self.work_img) #存储为图像
        plt.show()
        
    def obstacle2(self):
        x = 100
        y = 100
        rect_start = (x,y)
        x1 = 190
        y1 = 190
        rect_end = (x1,y1)
        cv2.rectangle(self.ori_img, rect_start, rect_end, (255,0,0), 60, 0)
        cv2.imshow('img',self.ori_img)
        cv2.imwrite('./obstacle2.jpg',self.work_img) #存储为图像
        if cv2.waitKey(40) & 0xFF == ord('q'): 
            return

    def process(self):
        self.obstacle()
        self.obstacle2()
if __name__ == '__main__':
    path = '../image/chao.jpg'
    aim = Detect(path)
    aim.process()
