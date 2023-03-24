import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

img = cv2.imread('ImagesProjetL3/30.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#bfilter = cv2.bilateralFilter(sobel, 11, 17, 17) #Noise reduction
_, binarizedImage = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
edged = cv2.Canny(binarizedImage, 30, 200) #Edge detection
img_dilate = cv2.dilate(edged, kernel, iterations=1)
contours, hierarchy = cv2.findContours(binarizedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

"""for cnt in contours:
   x1,y1 = cnt[0][0]
   approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
   if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(cnt)
      ratio = float(w)/h
      if ratio >= 0.9 and ratio <= 1.1:
         img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
         cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      else:
         cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
         img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
"""

cv2.drawContours(img, contours, -1, (0,255,0), 3)
# Réduire la taille de la fenêtre d'affichage
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 700)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()