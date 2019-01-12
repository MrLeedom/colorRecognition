'''
from PIL import Image

#打开一个jpg图像文件，注意是当前路径
im = Image.open('C:\\Users\\leedom\\Pictures\sample.jpg')
w, h = im.size
print('Original image size: %sx %s' % (w, h))
'''

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

#随机字母
def rndChar():
    return chr(random.randint(65, 90))

#随机颜色
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
    
#随机颜色
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))
    
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象:
font = ImageFont.truetype('C:\\Windows\\Fonts\\Arial.ttf', 36)
# 创建Draw对象:
draw = ImageDraw.Draw(image)
# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
# 模糊:
image = image.filter(ImageFilter.BLUR)
image.save('code.jpg', 'jpeg')