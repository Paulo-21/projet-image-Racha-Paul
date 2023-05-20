import cv2
import numpy as np
import matplotlib.pyplot as plt



# Ouvrir l'image
img = cv2.imread('ImagesProjetL3/image1.jpg')


# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Réduire la taille de la fenêtre d'affichage
cv2.namedWindow('Image Binarisée', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image Binarisée', 900, 900)

# Afficher l'image originale et l'image en niveaux de gris
#cv2.imshow('Image originale', img)
#cv2.imshow('Image en niveaux de gris', gray)


#Seuillage => image binarisée
_, binarizedImage = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)


#Histogramme de projection horizontale 
horizontal_projection = np.sum(binarizedImage, axis=1);



print(horizontal_projection);
plt.plot(horizontal_projection)
plt.show()
# Afficher l'image résultante
cv2.imshow('Image Binarisée', binarizedImage)

# Attendre que l'utilisateur appuie sur une touche pour fermer les fenêtres
cv2.waitKey(0)
cv2.destroyAllWindows()
