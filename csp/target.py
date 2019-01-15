# coding:UTF-8
'''
   @author:leedom

   Created on Tue Jan 15 09:47:46 2019
   description: 框住目标色块
'''
import cv2  # cv2是opencv官方的一个扩展库，里面含有各种有用的函数以及进程，opencv是一个基于开源发行的跨平台计算机视觉库，它轻量级而且高效
import numpy as np  # Numeric Python，它是由一个多维数组对象和用于处理数组的例程集合组成的库。Numpy拥有线性代数和随机数生成的内置函数
#对于数组的操作后期还要了解，几个维度？
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

    #颜色区域提取按矩阵思路
    def color_matrix(self):
        c = cv2.resize(self.ori_img, (819, 460))
        b = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        thresh = (thresh / 255).astype(np.int8)
        thresh = 1-thresh
        return thresh
        '''
        f = open('F:\\maomi\\test.txt','w')
        for i in thresh:
            for j in i:
                f.write(str(j)+'\r')
            f.write('\n')
        f.close()      
        return f
        '''

    def matrix_heart(self, text):
        #遍历矩阵文件中的每一行
        sum, sum_x, sum_y = 0, 0, 0
        a = -1
        for i in text:
            a = a + 1
            b = -1
            for j in i:
                b = b + 1
                sum_x = sum_x + b * int(str(j))
                sum_y = sum_y + a * int(str(j))
                sum = sum + int(str(j))
        key_point = (math.ceil(sum_x / sum), math.ceil(sum_y / sum))
        return key_point

    def neighbors(self, array, radius, x, y):
        arrayRsize, arrayCsize = len(array), len(array[0])
        pos = [[(n+x, i+y) for n in range(-1*radius, radius+1)]
               for i in range(-1*radius, radius+1)]
        min_distance = max(arrayCsize, arrayRsize)
        sum = 0

        def _getNum(rid, cid):
            if (rid < 0
                or cid < 0
                or arrayRsize <= cid
                    or arrayCsize <= rid):
                return 0
            else:
                if (array[cid, rid] == 0 or rid == 0 or cid == 0 or rid == arrayCsize - 1 or cid == arrayRsize - 1):
                    distance = math.ceil(np.sqrt(
                        np.square(x - rid) + np.square(y - cid)))
                else:
                    distance = -1
                return distance
        # [[_getNum(rid, cid) for rid, cid in row] for row in pos]

        #遍历所形成的坐标集
        for row in pos:
            for rid, cid in row:
                length = _getNum(rid, cid)
                if length < 0:
                    sum += 1
                elif length == 0:
                    print('fail')
                else:
                    min_distance = length
        if sum == np.square(radius * 2 + 1):  # 作为一个标志，表明此尺度并不是我们想要找的半径
            return 0
        return min_distance

    def four_points(self, distance, x, y):
        points = [[x-distance, y-distance], [x+distance, y-distance],
                  [x+distance, y+distance], [x-distance, y+distance]]
        points = np.array(points)
        return points

    # 画出关键轮廓的最校外接矩形

    def key_cnt_draw(self, points):
        mask = np.zeros(self.ori_img.shape, np.uint8)
        cv2.drawContours(mask, [points], -1, (255, 0, 0), thickness=2)
        return mask

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
        angle = math.atan(abs(points[0] - img_center_x) /
                          abs(points[1] - img_center_y))
        key = angle * 180 / math.pi
        print('角度是:', key)
        return key_img2

    def reality(self, w1, w2, distance):
        F = (30 * 3) / 0.3  # 人为定义
        real_distance = (F * 0.3) / w1
        edge = (F * w2) / w1
        angle = math.asin(edge / real_distance)
        key = angle * 180 / math.pi
        print('实际距离是：', real_distance)
        print('实际角度是：', angle)

    # 运行主函数
    def img_process_main(self):
        radius = 1
        text = self.color_matrix()
        key_point = self.matrix_heart(text)
        distance = self.neighbors(text, radius, key_point[0], key_point[1])
        while distance == 0:
            radius += 1
            distance = self.neighbors(
                text, radius + 1, key_point[0], key_point[1])
        points = self.four_points(distance, key_point[0], key_point[1])
        cnt_img = self.key_cnt_draw(points)
        key_distance = self.distance(points)
        aim_point = (key_distance[1], key_distance[2])
        aim_img = self.photo_Handle(cnt_img, aim_point)

        print('距离是：', key_distance[0])
     #   reality(w1,w2,key_distance[0])   实际的角度和距离
        cv2.imshow("images", np.hstack([self.ori_img, aim_img]))
        cv2.imshow("aimimg",aim_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()



if __name__ == '__main__':
    root_path = 'F:/maomi/'
    img_path = root_path + '/' + str(6) + '.png'
    d = Detect(img_path)
    d.img_process_main()
