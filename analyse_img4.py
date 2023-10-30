import cv2
import numpy as np

# Charger l'image
image = cv2.imread('1.jpg')

# Définir les bornes de la couleur rouge (dans l'espace de couleur BGR)
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])

# Convertir l'image en espace de couleur HSV (plus approprié pour la détection de couleurs)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Créer un masque pour les pixels rouges
orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Appliquer un flou pour éliminer le bruit
orange_mask = cv2.GaussianBlur(orange_mask, (5, 5), 0)

# Trouver les contours dans le masque rouge
contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialiser un compteur
n = 0

# Taille minimale d'une zone rouge
min_area = 50

# Analyser les contours détectés
for contour in contours:
    # Calculer la superficie de la zone entourée par le contour
    area = cv2.contourArea(contour)

    # Si la superficie est supérieure à la taille minimale, considérez cette zone comme une zone rouge
    if area >= min_area:
        n += 1

# Afficher le nombre de zones rouges détectées
print("Nombre de zones oranges détectées :", n)
