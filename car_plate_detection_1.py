import cv2 
#import matplotlib.pyplot as plt

car_plate_1_before = cv2.imread('car.jpg')
car_plate_1 = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE) #將圖片變灰

cv2.GaussianBlur(car_plate_1,(5,5),10) #將圖片進行高斯模糊 可以減少干擾
cv2.Sobel(car_plate_1,cv2.CV_8U,1,0,ksize=1) #將圖片反白
cv2.Canny(car_plate_1,250,100)

cv2.imshow('before', car_plate_1_before)
cv2.imshow('after', car_plate_1)
cv2.waitKey(0)
cv2.destroyAllWindows()