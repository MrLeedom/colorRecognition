#https://blog.csdn.net/setaire/article/details/78578506
#目标：利用python处理图片中的像素点，有时候在做验证码的时候，验证码一般都有干扰物，需要对验证码进行预处理
from PIL import Image
import itertools
img = Image.open('F:\\maomi\\0.png').convert('L')  # 打开图片,convert图像类型有L,RGBA

# 转化为黑白图


def blackWrite(img):
    blackXY = []

    # 遍历像素点
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            print(img.getpixel((x, y)))
            if img.getpixel((x, y)) < 128:
                img.putpixel((x, y), 0)  # 置为黑点
                blackXY.append((x, y))
            else:
                img.putpixel((x, y), 255)  # 置为白点
    return blackXY


# 去除干扰点
def clrImg(img, pointArr):
    # 获取周围黑点的个数
    def getN(p):
        count = 0
        x = [p[0]-1, p[0], p[0]+1]
        y = [p[1]-1, p[1], p[1]+1]
        for i in itertools.product(x, y):  # 笛卡尔积
            try:
                if img.getpixel(i) == 0:
                    count += 1
            except:
                print('out of')
                continue
        print(count)
        return count

    for p in pointArr:
        if getN(p) < 5:  # 周围黑点个数 <5 的黑点认为是干扰点,置为白点
            img.putpixel(p, 255)


pointArr = blackWrite(img)
clrImg(img, pointArr)
img.save("F:\\maomi\\aim1.png")
