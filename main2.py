import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils


img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer le filtre de Sobel
#Nous calculons le gradient dans les directions x et y en utilisant 1, 0 et 0, 1, la taille du noyau ksize=3.
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Combinez les résultats du filtre de Sobel x et y
sobel = np.sqrt(sobelx**2 + sobely**2)

# Normaliser les valeurs de pixel pour une meilleure visualisation
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#bfilter = cv2.bilateralFilter(sobel, 11, 17, 17) #Noise reduction
_, binarizedImage = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(sobel, 30, 200) #Edge detection
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

location
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]



# Réduire la taille de la fenêtre d'affichage
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 700)


#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
cv2.imshow('Image',edged)
cv2.imshow('c', cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()