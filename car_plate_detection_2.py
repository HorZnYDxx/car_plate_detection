import cv2 
import numpy as np
import matplotlib.pyplot as plt

f, arr = plt.subplots(2,3)

#原圖
car_plate_1 = cv2.imread('car.jpg')
arr[0,0].imshow(car_plate_1)


#灰圖
car_plate_2 = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE)
arr[0,1].imshow(car_plate_2)

#將圖片進行高斯模糊 可以減少干擾
car_plate_3 = cv2.GaussianBlur(car_plate_2,(5,5),10)
arr[0,2].imshow(car_plate_3)

#Sobel邊緣檢測
car_plate_4 = cv2.Sobel(car_plate_3,cv2.CV_8U,1,0,ksize=1) 
arr[1,0].imshow(car_plate_4)

#Laplacian邊緣檢測
car_plate_5 = cv2.Canny(car_plate_4,250,100)
arr[1,1].imshow(car_plate_5)

#二值化處理0~255
a,car_plate_6 = cv2.threshold(car_plate_5,0,255,cv2.THRESH_BINARY)
arr[1,2].imshow(car_plate_6)
print(car_plate_6.shape)

plt.show()
