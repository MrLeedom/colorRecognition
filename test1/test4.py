import numpy as np
import argparse
import cv2
import colorsys  # 提取图片中主要颜色
from PIL import Image  # python imaging library,已经是python平台事实上的图像处理标准库
import numpy as np
from skimage import draw


image = cv2.imread('F:\\maomi\\0.png')

def color_Handle():
    color = [
        # 黄色范围~这个是我自己试验的范围，可根据实际情况自行调整~注意：数值按[b,g,r]排布
    #   ([0, 70, 70], [100, 255, 255])
        ([0, 0, 200], [90, 80, 255])
    ]
    #如果color中定义了几种颜色区间，都可以分割出来
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应颜色
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # 展示图片
        cv2.imshow("images", np.hstack([image, output]))
#       cv2.imshow("haha",output)
        cv2.waitKey(0)
    return output




#提取图片中的主要颜色
def get_dominant_color(image):

    #颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')

#生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))

    max_score = 0  # 原来的代码此处为None
    dominant_color = 0  # 原来的代码此处为None，但运行出错，改为0以后 运行成功，原因在于在下面的 < span style = "font-family:Arial, Helvetica, sans-serif;" > score > max_score的比较中，max_score的初始格式不定 < /span >

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue

        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
            red = r
            green = g
            blue = b
      
    return dominant_color,red,green,blue


def area():
     sp = image.shape  # obtain the image shape
     sz1 = sp[0]  # height(rows) of image
     sz2 = sp[1]  # width(colums) of image
     x = sz1 / 2
     y = sz2 / 2
     

     rule = 2
     Y=np.array([a,b,a,b])
     X=np.array([c,c,d,d])
     rr, cc = draw.polygon(Y, X)
     draw.set_color(img, [rr, cc], [255, 0, 0]) #画出规则矩形

     #面积的计算
     area = (b - a) * (d - c)
     total_area = a * y
     distance = (area / total_area) * rule 
     cropImg = image[a:b, c:d]  # crop the image
     cv2.imwrite(dest, cropImg)  # write in destination path
     return distance


#找到四个目标点的函数
def aim_point(img):
    #遍历像素点
    x_min, y_min,x_max,y_max = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):           
            if (img[x,y,0] > 200):
                if x >= x_max :
                    x_max = x
                elif x <= x_min :
                    x_min = x
                else :
                    print('x no change') 

                if y >= y_max:
                    y_max = y
                elif y <= y_min:
                    y_min = y
                else:
                    print('y no change')
            else :
                print('black')
    A = (x_min,y_min)
    B = (x_max,y_min)
    C = (x_max,y_max)
    D = (x_min,y_max)
    return A,B,C,D

            



#找到距离中心点最近的函数
#def key_point(a,b,c,d):

color_Handle()




