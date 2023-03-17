import cv2 
import numpy as np
import matplotlib.pyplot as plt

#f, arr = plt.subplots(3,3)

#原圖
car_plate_1 = cv2.imread('car.jpg')
#arr[0,0].imshow(car_plate_1)


#灰圖
car_plate_2 = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE)
#arr[0,1].imshow(car_plate_2)

#將圖片進行高斯模糊 可以減少干擾
car_plate_3 = cv2.GaussianBlur(car_plate_2,(5,5),10)
#arr[0,2].imshow(car_plate_3)

#Sobel邊緣檢測
car_plate_4 = cv2.Sobel(car_plate_3,cv2.CV_8U,1,0,ksize=1) 
#arr[1,0].imshow(car_plate_4)

#Laplacian邊緣檢測
car_plate_5 = cv2.Canny(car_plate_4,250,100)
#arr[1,1].imshow(car_plate_5)

#二值化處理0~255
a,car_plate_6 = cv2.threshold(car_plate_5,0,255,cv2.THRESH_BINARY)
#arr[1,2].imshow(car_plate_6)

# 可以侵蝕和擴張
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(43,33))
car_plate_7 = cv2.dilate(car_plate_6,kernel)
#arr[2,0].imshow(car_plate_7)


# 迴圈找到所有的輪廓
A,B = cv2.findContours(car_plate_7,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
result = []
for i in A:
    x,y,w,h = cv2.boundingRect(i) #A是一堆輪廓的項目 回傳輪廓的左上角和長寬尺寸(左上角xy座標 長寬wh)
    if w>2*h:
        plt.imshow(car_plate_1[y:y+h,x:x+w]) #裁切出某張圖片的高度y到y+h 橫向位置座標為x到x+w
        plt.show()
        result.append(car_plate_1[y:y+h,x:x+w])


carPlate = result[2] # 讀取圖片
carPlateGray = cv2.cvtColor(carPlate, cv2.COLOR_BGR2GRAY)  # 轉換成灰圖
ret, carPlate_thre = cv2.threshold(carPlateGray, 100, 255, cv2.THRESH_BINARY_INV) #將灰度影像二值化，設定閾值是100

#分割字元
white = []
black = []
height = carPlate_thre.shape[0]
width = carPlate_thre.shape[1]
white_max = 0
black_max = 0
for i in range(width):
    s = 0  # 這一列白色總數
    t = 0  # 這一列黑色總數
    for j in range(height):
        if carPlate_thre[j][i] == 255:
            s += 1
        if carPlate_thre[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)

arg = False  # False表示白底黑字；True表示黑底白字
if black_max > white_max:
    arg = True

# 分割影像
def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):  # 0.95這個引數請多調整，對應下面的0.05（針對畫素分佈調節）
            end_ = m
            break
    return end_

n = 1
start = 1
end = 2
word = []
while n < width - 2:
    n += 1
    if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
        # 上面這些判斷用來辨別是白底黑字還是黑底白字
        # 0.05這個引數請多調整，對應上面的0.95
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            cj = carPlate[1:height, start:end]
            cj = cv2.resize(cj, (15, 30))
            word.append(cj)



print(len(word))
for i,j in enumerate(word):
    plt.subplot(1,8,i+1)
    plt.imshow(word[i],cmap='gray')

plt.show()
