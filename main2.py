import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

img = cv2.imread('ImagesProjetL3/image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#bfilter = cv2.bilateralFilter(sobel, 11, 17, 17) #Noise reduction
_, binarizedImage = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
edged = cv2.Canny(binarizedImage, 30, 200) #Edge detection
img_dilate = cv2.dilate(edged, kernel, iterations=1)
contours, hierarchy = cv2.findContours(binarizedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
# Réduire la taille de la fenêtre d'affichage
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 700)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()