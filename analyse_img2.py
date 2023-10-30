import cv2
import numpy as np

# Charger l'image
image = cv2.imread('1.jpg')

# Définir les bornes de la couleur rouge (dans l'espace de couleur BGR)
lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 50, 255])

# Convertir l'image en espace de couleur HSV (plus approprié pour la détection de couleurs)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Créer un masque pour les pixels rouges
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Appliquer un flou pour éliminer le bruit
red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)

# Trouver les contours dans le masque rouge
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialiser un compteur
n = 0

# Taille minimale d'une zone rouge
min_area = 2

# Analyser les contours détectés
for contour in contours:
    # Calculer la superficie de la zone entourée par le contour
    area = cv2.contourArea(contour)

    # Si la superficie est supérieure à la taille minimale, considérez cette zone comme une zone rouge
    if area >= min_area:
        n += 1

# Afficher le nombre de zones rouges détectées
print("Nombre de zones rouges détectées :", n)
