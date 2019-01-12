# coding:UTF-8
import cv2  
import numpy as np  
from skimage import draw
import math
import datetime
import time
F = 870                     #焦距,这个需要根据实际的公式计算得到
goods = 10                  #cm，实际物体的宽度
class Detect:
    def __init__(self, path):       
        self.ori_img = path
        self.gray = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2HSV)
        # 获得原始图像行列
        rows, cols = self.ori_img.shape[:2]
        # 工作图像
        self.work_img = cv2.resize(
            self.ori_img, (int(cols), int(rows)))
        self.work_gray = cv2.resize(
            self.gray, (int(cols), int(rows)))
        self.work_hsv = cv2.resize(
            self.hsv, (int(cols), int(rows)))

    # 颜色区域提取
    def color_area(self):
        #提取蓝色区域
        low_blue = np.array([100,43,46])
        high_blue = np.array([124,255,255])
        mask_blue = cv2.inRange(self.work_hsv,low_blue,high_blue)
        # 提取红色区域(暂定框的颜色为红色)
        low_red1 = np.array([0, 43, 46])
        high_red1 = np.array([8, 255, 255])
        # 将低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255
        mask1 = cv2.inRange(self.work_hsv, low_red1, high_red1)
        sum1 = np.sum(mask1)
        low_red2 = np.array([156, 43, 46])
        high_red2 = np.array([180, 255, 255])
        # 将低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255
        mask2 = cv2.inRange(self.work_hsv, low_red2, high_red2)
        sum2 = np.sum(mask2)
        if sum1>sum2:
            mask_red = mask1
        else:
            mask_red = mask2
        # mask = np.logical_or(mask_blue,mask_red) + 0
        # mask = mask.astype(np.uint8)
        # print(mask.dtype)
        # print(mask_red.dtype)
        #对图像进行与操作，并且有个掩膜的说法，得到更精确的图像轮廓
        blue = cv2.bitwise_and(self.work_hsv,self.work_hsv,mask = mask_blue)
        red = cv2.bitwise_and(self.work_hsv, self.work_hsv, mask=mask_red)
        # red = self.work_hsv * mask
        # cv2.imshow('blue',blue)
        # cv2.imshow('red',red)
        return red,blue  # 变成黑白图片
 
    # 形态学处理
    def good_thresh_img(self, img,blue):
        # hsv空间变换到gray空间,红色和蓝色
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blue_img = cv2.cvtColor(blue,cv2.COLOR_HSV2BGR)
        blue_img = cv2.cvtColor(blue,cv2.COLOR_BGR2GRAY)
        # 阈值处理
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _blue,blue_thresh = cv2.threshold(blue_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #做一些形态学操作,去一些小物体干扰，这是个边缘检测算法
        #img_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        #进行腐蚀膨胀，处理后的图片黑白分明，效果明显更优
        img_morph = cv2.dilate(thresh, se)
        img_morph = cv2.erode(img_morph, se)
        img_morph = cv2.erode(img_morph, se, iterations=1)
        img_morph = cv2.dilate(img_morph, se, iterations=1)

        blue_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        blue_morph = cv2.dilate(blue_thresh, blue_se)
        blue_morph = cv2.erode(blue_morph,blue_se)
        blue_morph = cv2.erode(blue_morph, blue_se, iterations = 1)
        blue_morph = cv2.dilate(blue_morph,blue_se, iterations = 1)
        # cv2.imshow('red',img_morph)
        return img_morph,blue_morph 

    # 矩形四角点提取,矩形的边缘点以及宽度反馈出来了
    def key_points_tap(self, img, blue):
        red_points = []
        red_widths = []
        blue_points = []
        blue_widths = []
        img_cp = img.copy()
        img_blue = blue.copy()
        # 按结构树模式找所有轮廓
        aa,cnts, _ = cv2.findContours(img_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bb,blues,blue_ = cv2.findContours(img_blue,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 按区域大小排序,找到第二大轮廓
        flag = 0
        if len(cnts) == 0 and len(blues) == 0:
            flag = 2       #两个异常
        elif len(cnts) !=0 and len(blues) != 0:
            flag = 1
            # cnt_list = sorted(cnts,key=cv2.contourArea, reverse=True)
            # cnt_list = []
            if len(cnts) >= 3:
                length = 3
            else:
                length = len(cnts)
            for i in range(length):
                cnt_one =sorted(cnts,key=cv2.contourArea, reverse=True)[i]
                box = cv2.minAreaRect(cnt_one)
                width = box[1][0]
                red_points.append(np.intc(cv2.boxPoints(box)))
                red_widths.append(width)
            # cnt_first = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            # cnt_second = sorted(cnts, key=cv2.contourArea, reverse=True)[1]
            # cnt_third = sorted(cnts, key=cv2.contourArea, reverse=True)[2]
            # box = cv2.minAreaRect(cnt_first)
            # box2 = cv2.minAreaRect(cnt_second)
            # box3 = cv2.minAreaRect(cnt_third)
            # width = box[1][0]
            # width2 = box2[1][0]
            # width3 = box3[1][0]
            # red_points.append(np.intc(cv2.boxPoints(box)))
            # red_points.append(np.intc(cv2.boxPoints(box2)))
            # red_points.append(np.intc(cv2.boxPoints(box3)))
            # red_widths.append(width)
            # red_points.append(width2)
            # red_points.append(width3)
            if len(blues) >= 3:
                blue_length = 3
            else:
                blue_length = len(blues)
            for i in range(blue_length):
                cnt_two =sorted(blues,key=cv2.contourArea, reverse=True)[i]
                blue_box = cv2.minAreaRect(cnt_two)
                blue_width = blue_box[1][0]
                blue_points.append(np.intc(cv2.boxPoints(blue_box)))
                blue_widths.append(blue_width)
            # blue_first = sorted(blues, key=cv2.contourArea, reverse=True)[0]
            # blue_second = sorted(blues, key= cv2.contourArea,reverse=True)[1]
            # blue_third = sorted(blues, key=cv2.contourArea, reverse = True)[2]
            # #找轮廓的最小外接矩形((point), (w, h))
            # blue_box = cv2.minAreaRect(blue_first)
            # blue_box2 = cv2.minAreaRect(blue_second)
            # blue_box3 = cv2.minAreaRect(blue_third)
            # blue_width = blue_box[1][0]
            # blue_width2 = blue_box2[1][0]
            # blue_width3 = blue_box3[1][0]
            # blue_points.append(np.intc(cv2.boxPoints(blue_box)))
            # blue_points.append(np.intc(cv2.boxPoints(blue_box2)))
            # blue_points.append(np.intc(cv2.boxPoints(blue_box3)))
            # blue_widths.append(blue_width)
            # blue_widths.append(blue_width2)
            # blue_widths.append(blue_width3)
        else:
            if len(cnts) != 0:
                if len(cnts) >= 3:
                    length = 3
                else:
                    length = len(cnts)
                for i in range(length):
                    cnt_one =sorted(cnts,key=cv2.contourArea, reverse=True)[i]
                    box = cv2.minAreaRect(cnt_one)
                    width = box[1][0]
                    red_points.append(np.intc(cv2.boxPoints(box)))
                    red_widths.append(width)
                    flag = 'blue'           
            if len(blues) != 0:
                if len(blues) >= 3:
                    blue_length = 3
                else:
                    blue_length = len(blues)
                for j in range(blue_length):
                    cnt_two =sorted(blues,key=cv2.contourArea, reverse=True)[j]
                    blue_box = cv2.minAreaRect(cnt_two)
                    blue_width = blue_box[1][0]
                    blue_points.append(np.intc(cv2.boxPoints(blue_box)))
                    blue_widths.append(blue_width)
                    flag = 'red'
        if flag == 2:
            return -1,-1,0,0,2
        elif flag == 'red':
            return -1,blue_points,0, blue_widths,'red'
        elif flag == 'blue':
            return red_points,-1,red_widths,0,'blue'
        else:
            return red_points,blue_points,red_widths,blue_widths,0

    # 画出关键轮廓的最校外接矩形,该函数的返回值貌似没啥用了，主要的画框已经在内部做完了
    def key_cnt_draw(self, points):
        mask = np.zeros(self.work_gray.shape, np.uint8)
        cv2.drawContours(self.work_img, [points], -1, 255, 2)
        return mask

    # 目标框图像中心点提取
    def center_point_cal(self, points):
        pt1_x, pt1_y = points[0, 0], points[0, 1]
        pt3_x, pt3_y = points[2, 0], points[2, 1]
        center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2
        return center_x, center_y

    def right_or_left(self,points,center):
        rows, cols = self.work_img.shape[:2]
        img_center_x, img_center_y = cols / 2, rows / 2
        if img_center_x >= center[0]:
            # print('right')
            return True
        else:
            # print('left')
            return False
        
  
    #算出目标点和四个关键点之间的最短距离
    def distance(self, points):
        # 得到图像中心
        rows, cols = self.work_img.shape[:2]
        img_center_x, img_center_y = cols / 2, rows / 2
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

    #计算两个中心点之间的距离
    def centerDistance(self,center1,center2):
        distance = np.sqrt(np.square(center1[0] - center2[0]) + np.square(center1[1] - center2[1]))
        return distance

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
        return key_img2

    def get_Angle_Distance(self, width, gap, flag):
        width = (width / 5) * 4
        distance = F * goods / width
        pre_distance = gap * goods / width
        ratio = pre_distance / distance
        if ratio >= -1 and ratio <= 1:
            angle = math.asin(pre_distance / distance)
            result = angle * 180 / math.pi
            if flag:
                return distance,result 
            else:
                return distance,(-1)*result
        else:
            return -1,-1

    # 运行主函数
    def img_process_main(self,count):
        prior = 3
        start = datetime.datetime.now()   
        # 找到红色区域
        red, blue = self.color_area()
        # cv2.imshow('red',red)
        # 处理得到一个比较好的二值图
        img_morph, blue_morph = self.good_thresh_img(red,blue)
        # 获取矩形框的四个关键点
        red_points,blue_points,red_width, blue_width,flag= self.key_points_tap(img_morph, blue_morph)
        
        min_distance = 0
        num1 = 0
        num2 = 0
        key_length = 0  #距离初始化为０
        key_angle = 0  #角度初始化为０
        good = False
        if flag == 2:
            pass
        elif flag == 'blue':       
            red_center = self.center_point_cal(red_points[0])          #红色区域中心点
            # print('红色中心点：',red_center)
            red_flag_right_left = self.right_or_left(red_points[0],red_center)
            red_distance = self.distance(red_points[0])
            # print('图片中物体的像素宽度：',red_width)
            cv2.drawContours(self.work_img, [red_points[0]], -1, (0,0,255), 2)
            red_length,red_angle = self.get_Angle_Distance(red_width[0],red_distance[0],red_flag_right_left)
            text1 = 'red_length:' + str(round(red_length,2))
            cv2.putText(self.work_img, text1, (self.work_img.shape[1]-450,self.work_img.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            text2 = "red_angle:" + str(round(red_angle,2))
            cv2.putText(self.work_img, text2, (self.work_img.shape[1]-400,self.work_img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),3)
        elif flag == 'red':
            blue_center = self.center_point_cal(blue_points[0])
            # print('蓝色中心点:',blue_center)
            blue_flag_right_left = self.right_or_left(blue_points[0],blue_center)
            blue_distance = self.distance(blue_points[0])
            cv2.drawContours(self.work_img, [blue_points[0]], -1, (255,0,0), 2)
            blue_length,blue_angle = self.get_Angle_Distance(blue_width[0],blue_distance[0],blue_flag_right_left)
            text1 = 'blue_length:' + str(round(blue_length,2))
            cv2.putText(self.work_img, text1, (self.work_img.shape[1]-450,self.work_img.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            text2 = "blue_angle:" + str(round(blue_angle,2))
            cv2.putText(self.work_img, text2, (self.work_img.shape[1]-400,self.work_img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),3)
        else:
            for i in range(len(red_points)):
                for j in range(len(blue_points)):
                    two_length = red_width[i] / 2 + blue_width[j] / 2
                    red_center = self.center_point_cal(red_points[i])
                    blue_center = self.center_point_cal(blue_points[j])
                    # print('进入第三种判断')
                    red_flag = self.right_or_left(red_points,red_center)
                    blue_flag = self.right_or_left(blue_points, blue_center)
                    red_distance = self.distance(red_points[i])
                    blue_distance = self.distance(blue_points[j])
                    red_flag_right_left = self.right_or_left(red_points[i],red_center)
                    red_length,red_angle = self.get_Angle_Distance(red_width[i],red_distance[0],red_flag_right_left)
                    # print('red_length',red_length)
                    # print('red_angle',red_angle)
                    blue_flag_right_left = self.right_or_left(blue_points[j],blue_center)
                    blue_length,blue_angle = self.get_Angle_Distance(blue_width[j],blue_distance[0],blue_flag_right_left)
                    # print('blue_length',blue_length)
                    # print('blue_angle',blue_angle)
                
                    if red_width[i] > 2*blue_width[j] :
                        # cv2.drawContours(self.work_img, [red_points[i]], -1, (0,0,255), 2)
                        text2 = 'only red'
                        # cv2.putText(self.work_img, text2, (self.work_img.shape[1]-300,self.work_img.shape[0]-120),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),3)             
                    elif blue_width[j] > 2*red_width[i]:
                        # cv2.drawContours(self.work_img, [[blue_points[j]]], -1,(255,0,0),2)
                        text2 = 'only blue'
                        # cv2.putText(self.work_img, text2, (self.work_img.shape[1]-300,self.work_img.shape[0]-120),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,0,0),3)
                    else:
                        #需要判断两个点之间的距离关系
                        aim_distance = self.centerDistance(red_center, blue_center)
                        if i ==0 and j == 0:
                            min_distance = aim_distance
                            num1 = i
                            num2 = j
                            if aim_distance < 1.2 * two_length:
                                good = True
                        if aim_distance < 1.2*two_length and aim_distance < min_distance :
                            good = True
                            min_distance = aim_distance
                            num1 = i
                            num2 = j
            if num1 == -1 or num2 == -1:
                print('说明只有某一种情况,显示未找到临近的红蓝色块')
                cv2.putText(self.work_img, 'no aim', (self.work_img.shape[1]-300,self.work_img.shape[0]-120),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,0,0),3)
            else:
                if good == False:
                    cv2.putText(self.work_img, 'no aim', (self.work_img.shape[1]-300,self.work_img.shape[0]-120),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,0,0),3)
                else:
                    #找到了最近的红色区块和蓝色区块
                    two_length = red_width[i] / 2 + blue_width[j] / 2
                    red_center = self.center_point_cal(red_points[i])
                    blue_center = self.center_point_cal(blue_points[j])
                    # print('进入第三种判断')
                    red_flag = self.right_or_left(red_points,red_center)
                    blue_flag = self.right_or_left(blue_points, blue_center)
                    red_distance = self.distance(red_points[i])
                    blue_distance = self.distance(blue_points[j])
                    red_flag_right_left = self.right_or_left(red_points[i],red_center)
                    red_length,red_angle = self.get_Angle_Distance(red_width[i],red_distance[0],red_flag_right_left)
                    # print('red_length',red_length)
                    # print('red_angle',red_angle)
                    blue_flag_right_left = self.right_or_left(blue_points[j],blue_center)
                    blue_length,blue_angle = self.get_Angle_Distance(blue_width[j],blue_distance[0],blue_flag_right_left)
                    max_x = -1
                    max_y = -1
                    array1 = []
                    array2 = []
                    array3=[]
                    array4 = []               
                    for item in red_points[i]:
                        array1.append(item[0])
                        array2.append(item[1])
                    for item2 in blue_points[j]:
                        array1.append(item2[0])
                        array2.append(item2[1])
                    array1.sort()
                    array2.sort()            
                    array = []
                    x_min = array1[0]
                    y_min = array2[0]
                    x_max = array1[-1]
                    y_max = array2[-1]
                    max1 = []
                    max1.append(x_min)
                    max1.append(y_min)
                    array.append(max1)
                    max2 = []
                    max2.append(x_min)
                    max2.append(y_max)
                    array.append(max2)
                    max4 = []
                    max4.append(x_max)
                    max4.append(y_max)
                    array.append(max4)
                    max3 = []
                    max3.append(x_max)
                    max3.append(y_min)
                    array.append(max3)
                    # print('red_length')
                    key_center = self.center_point_cal(np.array(array))
                    key_position = self.right_or_left(np.array(array),key_center)
                    key_length,key_angle = self.get_Angle_Distance(blue_width[j]+red_width[i],blue_distance[0]+red_distance[0],key_position)
                    
                    cv2.drawContours(self.work_img, [red_points[i]], -1, (0,0,255), 2)
                    cv2.drawContours(self.work_img, [blue_points[j]], -1,(255,0,0),2)
                    cv2.drawContours(self.work_img, [np.array(array)], -1,(0,255,0),2)
                   
        # # 显示图像        1
        cv2.imshow('ori', self.work_img)
        # cv2.imwrite('./img/'+str(count) + '.jpg',self.work_img) #存储为图像
        # print(self.work_img.shape)  
        end = datetime.datetime.now()
        time = (end - start).microseconds
        print('time:%dus'%(time))
        return key_length,key_angle

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    root_path = '../image'    #实际图片存放的位置
    img_path = root_path + '/chao.jpg'
    d = Detect(img_path)
    # start = datetime.datetime.now()
    d.img_process_main()
    # end = datetime.datetime.now()
    # print((start-end).seconds)

'''
cv2是opencv官方的一个扩展库，里面含有各种有用的函数以及进程，opencv是一个基于开源发行的跨平台计算机视觉库，
它轻量级而且高效
Numeric Python，它是由一个多维数组对象和用于处理数组的例程集合组成的库。Numpy拥有线性代数和随机数生成的内置函数
#cv2.imread(文件名，属性) 读入图像
#cv2.imshow(窗口名，图像文件) 显示图像
#cv2.cvtColor颜色空间转换函数
#cv2.resize图像缩放函数

# 原始图像信息 self.ori_img = cv2.imread(path)

# aim_point = (distance[1],distance[2])       #得到离中心最近的点

# 找到矩形中心点
# rect_center_point = self.center_point_cal(points)
# 画出关键轮廓（调试用,并没有什么卯月）
# cnt_img = self.key_cnt_draw(points)
# aim_img = self.photo_Handle(cnt_img,aim_point)
'''