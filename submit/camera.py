import cv2 
import numpy as np 
import sample

def redRecognition():
    cap = cv2.VideoCapture(0) 
    count = 1     #计数器
    while(1): # get a frame 
        ret,frame = cap.read()
        d = sample.Detect(frame)
        d.img_process_main(count)
        count += 1
        #waitKey(int delay)其中delay<=0时表示无限期等待,而delay>0是表示等待的毫秒数
        #后者表示按q键终止程序
        if cv2.waitKey(40) & 0xFF == ord('q'): 
            break
    cap.release() 
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    redRecognition()

