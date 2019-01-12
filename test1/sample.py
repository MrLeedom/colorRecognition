# coding:UTF-8
import cv2  # cv2是opencv官方的一个扩展库，里面含有各种有用的函数以及进程，opencv是一个基于开源发行的跨平台计算机视觉库，它轻量级而且高效
import numpy as np  # Numeric Python，它是由一个多维数组对象和用于处理数组的例程集合组成的库。Numpy拥有线性代数和随机数生成的内置函数
from skimage import draw
import math
#cv2.imread(文件名，属性) 读入图像
#cv2.imshow(窗口名，图像文件) 显示图像
#cv2.cvtColor颜色空间转换函数
#cv2.resize图像缩放函数


class Detect:
    def __init__(self, path):
        # 原始图像信息
        self.ori_img = cv2.imread(path)
        self.gray = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2HSV)
        # 获得原始图像行列
        rows, cols = self.ori_img.shape[:2]

        #print(rows)
        #print(cols)
        # 工作图像
        self.work_img = cv2.resize(
            self.ori_img, (int(cols / 4)*5, int(rows / 4)*5))
        self.work_gray = cv2.resize(
            self.gray, (int(cols / 4)*5, int(rows / 4)*5))
        self.work_hsv = cv2.resize(
            self.hsv, (int(cols / 4)*5, int(rows / 4)*5))

    # 颜色区域提取
    def color_area(self):
        # 提取红色区域(暂定框的颜色为红色)
        low_red = np.array([156, 43, 46])
        high_red = np.array([180, 255, 255])
        # 将低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255
        mask = cv2.inRange(self.work_hsv, low_red, high_red)
        red = cv2.bitwise_and(self.work_hsv, self.work_hsv, mask=mask)
        return red  # 变成黑白图片
 
    # 形态学处理
    def good_thresh_img(self, img):
        # hsv空间变换到gray空间
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 阈值处理
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 做一些形态学操作,去一些小物体干扰
        img_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3))
        cv2.erode(img_morph, (3, 3), img_morph, iterations=2)
        cv2.dilate(img_morph, (3, 3), img_morph, iterations=2)
        return img_morph #处理后的图片黑白分明，效果明显更优

    # 矩形四角点提取
    def key_points_tap(self, img):
        img_cp = img.copy()
        # 按结构树模式找所有轮廓
        aa,cnts, _ = cv2.findContours(img_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 按区域大小排序,找到第二大轮廓
        cnt_second = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # 找轮廓的最小外接矩形((point), (w, h))
        box = cv2.minAreaRect(cnt_second)
        # ->(points)->(l_ints)
        return np.int0(cv2.boxPoints(box))

    # 画出关键轮廓的最校外接矩形

    def key_cnt_draw(self, points):
        mask = np.zeros(self.work_gray.shape, np.uint8)
        cv2.drawContours(mask, [points], -1, 255, 2)
        return mask

    # 目标框图像中心点提取
    def center_point_cal(self, points):
        pt1_x, pt1_y = points[0, 0], points[0, 1]
        pt3_x, pt3_y = points[2, 0], points[2, 1]
        center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2
        return center_x, center_y

  
    #算出目标点和四个关键点之间的最短距离

    def distance(self, points):
        # 得到图像中心
        rows, cols = self.work_img.shape[:2]
        img_center_x, img_center_y = cols / 2, rows / 2
        #得到四个点,下面四行代码无丝毫意义
        pt1_x, pt1_y = points[0, 0], points[0, 1]
        pt2_x, pt2_y = points[1, 0], points[1, 1]
        pt3_x, pt3_y = points[2, 0], points[2, 1]
        pt4_x, pt4_y = points[3, 0], points[3, 1]
        distance = 0
        for i in range(0, len(points)):
            a, b = points[i, :2]
            now_distance = np.sqrt(
                np.square(a - img_center_x) + np.square(b - img_center_y))
            if i == 0:
                distance = now_distance
                key_x = a
                key_y = b
            elif now_distance < distance:
                distance = now_distance
                key_x = a
                key_y = b
            else:
                print('no change')
        return distance, key_x, key_y

    #图片建系处理，以及连线
    def photo_Handle(self, img, points):
        # 得到图像中心
        rows, cols = img.shape[:2]
        img_center_x, img_center_y = int(cols / 2), int(rows / 2)
        x = [img_center_x, points[0]]
        y = [img_center_y, points[1]]
        key_img = cv2.line(img, (img_center_x, img_center_y),
                           (points[0], points[1]), 128, 2)
        key_img1 = cv2.line(key_img, (img_center_x, 0),
                            (img_center_x, rows), 128, 1)
        key_img2 = cv2.line(key_img1, (0, img_center_y),
                            (cols, img_center_y), 128, 1)
        # key_img = cv2.line(img,(0,0),(512,512),127,3)
        # rr, cc = draw.line(int(img_center_x), int(img_center_y), points[0], points[1])
        # draw.set_color(self.work_img, [rr, cc], [255, 0, 0])
        angle = math.atan(abs(points[0] - img_center_x) /
                          abs(points[1] - img_center_y))
        key = angle * 180 / math.pi
        print('角度是:', key)
        return key_img2

    # 运行主函数
    def img_process_main(self):
        
        # 找到红色区域
        red = self.color_area()
        # 处理得到一个比较好的二值图
        img_morph = self.good_thresh_img(red)
        # 获取矩形框的四个关键点
        points = self.key_points_tap(img_morph)
        distance = self.distance(points)
        print('距离是：',distance[0])
        aim_point = (distance[1],distance[2])

   
        # 找到矩形中心点
        rect_center_point = self.center_point_cal(points)
        # 画出关键轮廓（调试用,并没有什么卯月）
        cnt_img = self.key_cnt_draw(points)
        aim_img = self.photo_Handle(cnt_img,aim_point)

        
        # 显示图像
        cv2.imshow('ori', self.ori_img)
        cv2.imshow('red',red)
        cv2.imshow("images",  aim_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    root_path = 'F:/maomi/'    #实际图片存放的位置
    img_path = root_path + '/' + str(8) + '.png'
    d = Detect(img_path)
    d.img_process_main()
