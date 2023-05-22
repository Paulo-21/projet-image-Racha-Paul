import cv2
from matplotlib import pyplot as plt
import numpy as np
from itertools import groupby



img = cv2.imread('ImagesProjetL3/28.jpg')
kernel = np.ones((5,5), np.uint8)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#bfilter = cv2.bilateralFilter(sobel, 11, 17, 17) #Noise reduction
_, binarizedImage = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

#kernel = np.ones((5,5), np.uint8)
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
"""
# Réduire la taille de la fenêtre d'affichage
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 700)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


def segmentation_et_traitement_image(chemin_image: str):
    """
    Pour l'instant cette fonction ne fonctionne qu'avec les tableaux avec un fond vert(tableaux classique)
    Exemple: 'ImagesProjetL3/11.jpg', 'ImagesProjetL3/16.jpg', 'ImagesProjetL3/30.jpeg'...
    Retourne une liste  avec 2 éléments(images):
    Element 1--> isolation du tableau en couleur avec un fond noir
    Element 2 --> Image complétement noir avec en blanc  uniquement l'ecriture manuscrite ce qu'il y avait écrit
    sur le plateau
    :param chemin_image: chemin de l'image à traiter
    :return: la liste
    """
    # recupération image
    img = cv2.imread(chemin_image)

    # masque noir demême taille que l'image
    mask = np.zeros_like(img)

    # Conversion en niveaux de gris
    image_en_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarisation de l'image (en nuance de gris)
    binarisation = cv2.threshold(image_en_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # masque morphologique pour éroder puis dilater l'image binarisée (ouverture)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # erosion de l'image
    image_erode = cv2.erode(binarisation, kernel, iterations=1)

    # dilatation pour renforcer les contours
    image_erode_puis_dilate = cv2.dilate(image_erode, kernel, iterations=1)

    # Trouver les contours de l'image binarisée apres ouverture
    contours, hierarchy = cv2.findContours(image_erode_puis_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image d'origine en vert (r= 0, v=255, b= 0)
    cv2.drawContours(image_erode_puis_dilate, contours, -1, (0, 255, 0), 3)

    # On re-effectue le processus d'ouverture afin de réduire le bruit au maximum
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(image_erode_puis_dilate, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    # On a réduit au maximum  le bruit  au tour du tableau
    # Le pré-traitement de l'image est terminé

    # Recherche des contours
    contours2, hierarchy2 = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # A ce stade l'image ressemble à une image binarisée avec différentes formes et
    # surtout avec une forme (plutôt rectangulaire mais surtout PLUS GRANDE QUE LES AUTRES)
    # centrée dans lequel se trouve des écritures manuscrites noirs

    # Recherche de la forme avec la plus grande aire
    max_area = 0
    best_cnt = None
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    # L'image ayant été modifié par les traitements précédents, on recharge l'image originale
    img = cv2.imread(chemin_image)

    # On dessine les contours [en blanc (r=255,v=255,b=255)] de la forme avec la plus grande aire(le tableau)
    # sur le masque noir de même dimension que l'image originale
    cv2.drawContours(mask, [best_cnt], 0, (255, 255, 255), -1)

    # "Ou exclusif" entre le masque et l'image originale
    result = cv2.bitwise_and(img, mask)

    # nuance_de_gris
    gray_final = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # binarisation
    ret, thresh_final = cv2.threshold(gray_final, 145, 255, cv2.THRESH_BINARY)

    segmentation_tableau_image = result

    traitement_final_de_l_image = thresh_final

    return [segmentation_tableau_image, traitement_final_de_l_image]

#test fonction 'segmentation_et_traitement_image'

list_image = segmentation_et_traitement_image('ImagesProjetL3/image1.jpg')

img_original = cv2.imread('ImagesProjetL3/image1.jpg')

#cv2.namedWindow('Texte redressé', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Texte redressé', 1000, 700)
#cv2.namedWindow('Image segmentee', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Image segmentee', 1000, 700)
#cv2.namedWindow('Image traite', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Image traite', 1000, 700)
#cv2.namedWindow('Lines', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Lines', 1000, 700)
#**************************************************************************************************
#****************************************************************************************************
# Convertir en niveaux de gris
gray = cv2.cvtColor(list_image[0], cv2.COLOR_BGR2GRAY)

#contours
edges = cv2.Canny(gray, 50, 150, apertureSize=3)


#Detection de lignes en utilisant la transformation de hough 
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)

#calculer l'angle de chaque ligne 
angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    angles.append(angle)

#Calculer la moyenne de l'angle 
median_angle = np.median(angles)

# vérifier l'angle
if median_angle > 45:
    median_angle -= 180
elif median_angle < -45:
    median_angle += 180

# Corriger l'inclinaison 
(h, w) = list_image[0].shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
rotated = cv2.warpAffine(list_image[0], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#*******************************************************************************
#Extraction de lignes 
#*******************************************************************************
grayRotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)


img_inverted = cv2.bitwise_not(grayRotated)



_, binarizedRotated = cv2.threshold(img_inverted, 125, 255, cv2.THRESH_BINARY)


kernel = np.ones((4,4),np.uint8)
dilated = cv2.erode(binarizedRotated, kernel, iterations = 1)


#l'histogramme 
hist = cv2.reduce(dilated, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)


#detecter les ignes
lines = []
threshold = 760000  # seuil
for i in range(hist.shape[0]):
    if hist[i] < threshold:
        lines.append(i)

# Segmentation des lignes
height, width = dilated.shape
line_image = []
for _, rows in groupby(enumerate(lines), lambda x: x[1] - x[0]):
    rows = list(rows)
    line_img = dilated[rows[0][1]:rows[-1][1], 0:width]
    #vérifier que l'image n'est pas vide 
    if line_img.size > 0:  
        line_image.append(line_img)

'''
#affichage de chaque ligne 
for i, line in enumerate(line_image):
    line_resized = cv2.resize(line, (1000, 100))  # resize to width=500, height=100
    cv2.imshow(f'Line {i+1}', line_resized)
    cv2.waitKey(0)
'''





#cv2.imshow('Texte redresse',dilated)
#cv2.imshow('Image segmentee',list_image[0])
#cv2.imshow('Image traite',list_image[1])
#cv2.imshow('Image Original',img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()