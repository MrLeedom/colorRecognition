import os    #os包包括各种各样的函数，以实现操作系统的许多功能，os包的一些命令就是用于文件管理 
import cv2   #cv2是opencv官方的一个扩展库，里面含有各种有用的函数以及进程，opencv是一个基于开源发行的跨平台
#计算机视觉库，它轻量级而且高效--有 
import colorsys  #提取图片中主要颜色
from PIL import Image  #python imaging library,已经是python平台事实上的图像处理标准库

# 遍历指定目录，显示目录下的所有文件名  
def CropImage4File(filepath,destpath):  
    pathDir =  os.listdir(filepath) # list all the path or file  in filepath  
    for allDir in pathDir:  
        child = os.path.join(filepath, allDir)  
        dest = os.path.join(destpath,allDir)  
        if os.path.isfile(child): 
# judge whether is the dir or file
#下面写你需要进行的操作，注意缩进 
           image = cv2.imread(child)   
           sp = image.shape#obtain the image shape  
           sz1 = sp[0]#height(rows) of image  
           sz2 = sp[1]#width(colums) of image  
           #sz3 = sp[2]#the pixels value is made up of three primary colors, here we do not use  
           #你想对文件的操作  
           a=int(sz1/2-64) # x start  
           b=int(sz1/2+64) # x end  
           c=int(sz2/2-64) # y start  
           d=int(sz2/2+64) # y end  
           cropImg = image[a:b,c:d] #crop the image  
           cv2.imwrite(dest,cropImg)# write in destination path  


#提取图片中的主要颜色



def get_dominant_color(image):

    #颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')

#生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))

    max_score = 0 #原来的代码此处为None
    dominant_color = 0 #原来的代码此处为None，但运行出错，改为0以后 运行成功，原因在于在下面的 < span style = "font-family:Arial, Helvetica, sans-serif;" > score > max_score的比较中，max_score的初始格式不定 < /span >

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

    return dominant_color
             
if __name__ == '__main__':  
    filepath ='F:\\maomi' # source images  
    destpath='F:\\maomi_resize' # resized images saved here  
    CropImage4File(filepath,destpath) 
    print(get_dominant_color(Image.open('F:\\maomi\\blue.jpg')))
