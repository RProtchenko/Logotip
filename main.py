import cv2
import numpy
import numpy as np
import imutils
from matplotlib import  pyplot as pl
#Ввод нужных нам библиотек


img = cv2.imread("7.jpg")#Выбираем нужное изображение
data = np.array([[0,0]])



ct = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#меняем формат изображения
low = numpy.array([13,13,13])
high = numpy.array([256, 256, 256])
mask1= cv2.inRange(ct, low, high)#находим все цветные объекты нашего изображения
res = cv2.bitwise_and(img, img, mask =mask1)

gray = cv2.cvtColor(res , cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 300)# Включаем фильтры

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))#операция морфологии
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


cont = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# Записываем контуры
cont = imutils.grab_contours(cont)


con = 0
for c in cont:
    p = cv2.arcLength(c, True)
    approx= cv2.approxPolyDP(c, 0.02*p, True)# Приближение контура к другой форме

    if len(approx) > 6 and  len(approx)<9 : # Фильтруем длины наших контуров
        p = approx
        for c in p:
            ras = p
        min, max = np.min(ras,0),np.max(ras,0)# Находим максимально и минимальное значение контуров
        ras = np.array(max)-np.array(min)#Находим разность этих коэффициентов
        data = np.vstack([data, ras])#Запишим наши коэфиценты
        con += 1
        if data[con][0] > 75 and data[con][1] > 43 :#По коэффицинтам сравниваем нужные нам контуры
            s = approx


mask = np.zeros(gray.shape, np.uint8)#Создаем новое изображение
new_img =cv2.drawContours(mask,[s], 0,255, -1)# Рисуем контур на новом изображении
bitw_img = cv2.bitwise_and(img, img , mask = mask)#наносим нашу 'mask'на начальное изображение


x, y = np.where(mask == 255)
x1, y1 = np.min(x),np.min(y) #Вырезаем наш логотип
x2, y2 = np.max(x),np.max(y)
x1-=2
x2+=2
y2+=2
y1-=5
crop = img[x1:x2, y1:y2]

pl.imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
pl.show() #Выводим наш вырезанный логотип
x1-=3
x2+=3
y2+=4
y1-=5
cv2.rectangle(img,(y1,x1),(y2,x2), (0,0,255), thickness = 3) #Рисуем контур на нашем начальном изображении
cv2.imwrite("Result.jpg", img) #Записываем результат
