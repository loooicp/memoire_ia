import cv2
import numpy as np

# Charger l'image
image = cv2.imread('1.jpg')

# Définir les bornes de la couleur rouge (dans l'espace de couleur BGR)
lower_red = np.array([9, 155, 255])
upper_red = np.array([100, 255, 178])

# Convertir l'image en espace de couleur HSV (plus approprié pour la détection de couleurs)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Créer un masque pour les pixels rouges
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Appliquer un flou pour éliminer le bruit
red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)

# Initialiser un compteur
n = 0

# Taille des zones à analyser (5x5 pixels)
zone_size = 5

# Analyser l'image pixel par pixel
for y in range(0, image.shape[0], zone_size):
    for x in range(0, image.shape[1], zone_size):
        # Extraire la zone 5x5 pixels
        roi = red_mask[y:y+zone_size, x:x+zone_size]

        # Compter le nombre de pixels rouges dans la zone
        red_pixels = np.count_nonzero(roi)

        # Si le nombre de pixels rouges dépasse un seuil, considérez cette zone comme une zone rouge
        seuil = zone_size * zone_size * 0.6
        if red_pixels > seuil:
            n += 1

# Afficher le nombre de zones rouges détectées
print("Nombre de zones rouges détectées :", n)
