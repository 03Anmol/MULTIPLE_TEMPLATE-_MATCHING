import cv2 as cv
import numpy as np

img_rgb = cv.imread(r'C:\Users\Lenovo\Pictures\video\image0.jpg')
assert img_rgb is not None, "File could not be read, check the file path"

img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

temp = cv.imread(r'C:\Users\Lenovo\Pictures\video\croped.jpg', cv.IMREAD_GRAYSCALE)
temp2 = cv.imread(r'C:\Users\Lenovo\Pictures\video\crop2.jpg', cv.IMREAD_GRAYSCALE)
temp3 = cv.imread(r'C:\Users\Lenovo\Pictures\video\crop3.jpg', cv.IMREAD_GRAYSCALE)
temp4 = cv.imread(r"C:\Users\Lenovo\Pictures\video\crop4.jpg",cv.IMREAD_GRAYSCALE)

assert temp is not None
assert temp2 is not None
assert temp3 is not None

w, h = temp.shape[::-1]
w2, h2 = temp2.shape[::-1]
w3, h3 = temp3.shape[::-1]
w4,h4 = temp4.shape[::-1]

result_img = img_rgb.copy()

res = cv.matchTemplate(img_gray, temp, cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

res2 = cv.matchTemplate(img_gray, temp2, cv.TM_CCOEFF_NORMED)
threshold2 = 0.8
loc2 = np.where(res2 >= threshold2)
for pt2 in zip(*loc2[::-1]):
    cv.rectangle(result_img, pt2, (pt2[0] + w2, pt2[1] + h2), (0, 0, 255), 2)

res3 = cv.matchTemplate(img_gray, temp3, cv.TM_CCOEFF_NORMED)
threshold3 = 0.8
loc3 = np.where(res3 >= threshold3)
for pt3 in zip(*loc3[::-1]):
    cv.rectangle(result_img, pt3, (pt3[0] + w3, pt3[1] + h3), (255, 0, 0), 2)

res4 = cv.matchTemplate(img_gray,temp4,cv.TM_CCOEFF_NORMED)
threshold4 = 0.8
loc4 = np.where(res4>=threshold4)
for pt4 in zip(*loc4[::-1]):
    cv.rectangle(result_img, pt3, (pt4[0] + w4, pt4[1] + h4), (255, 0, 0), 2)


cv.imwrite('res.png', result_img)
cv.imshow('Result', result_img)
cv.waitKey(0)
cv.destroyAllWindows()
