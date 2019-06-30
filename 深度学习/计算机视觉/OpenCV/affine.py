#原文：https://blog.csdn.net/windowsyun/article/details/78158747 

import cv2
import numpy as np
img = cv2.imread('face.png')
rows, cols, ch = img.shape
 
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
 
M = cv2.getAffineTransform(pts1, pts2)
print(M)
dst = cv2.warpAffine(img, M, (cols, rows))


img1 = img.reshape((rows*2, cols//2,ch))
rows1, cols1, ch1 = img1.shape
print(img1.shape)
dst1 = cv2.warpAffine(img1, M, (cols1, rows1))

dst1 = dst1.reshape((rows, cols, ch))
cv2.imshow("img1",img1)
cv2.imshow('image', dst)
cv2.imshow('image1', dst1)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('affine.jpg', dst)
    cv2.imwrite('affine1.jpg', dst1)
    cv2.destroyAllWindows()





