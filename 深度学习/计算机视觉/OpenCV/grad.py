#原文：https://blog.csdn.net/Mr_zhuzj/article/details/82222107 


import cv2
import numpy as np
img=cv2.imread("face.png",cv2.IMREAD_UNCHANGED)
sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
sobelx=cv2.convertScaleAbs(sobelx)
sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
sobely=cv2.convertScaleAbs(sobely)
result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
 
scharrx=cv2.Scharr(img,cv2.CV_64F,dx=1,dy=0)
scharrx=cv2.convertScaleAbs(scharrx)
scharry=cv2.Scharr(img,cv2.CV_64F,dx=0,dy=1)
scharry=cv2.convertScaleAbs(scharry)
result1=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)


cv2.imshow("original",img)
cv2.imshow("result",result)
cv2.imshow("result1",result1)
cv2.waitKey()
cv2.destroyAllWindows()
