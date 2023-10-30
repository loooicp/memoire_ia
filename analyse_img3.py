import cv2
import numpy as np

# Charger l'image
image = cv2.imread('1.jpg')

# Définir les bornes des couleurs rouges et oranges (dans l'espace de couleur BGR)
lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 50, 255])

lower_orange = np.array([0, 100, 200])
upper_orange = np.array([50, 200, 255])

# Convertir l'image en espace de couleur HSV (plus approprié pour la détection de couleurs)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Créer des masques pour les pixels rouges et oranges
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Combinez les masques rouge et orange
combined_mask = cv2.bitwise_or(red_mask, orange_mask)

# Appliquer un flou pour éliminer le bruit
combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

# Trouver les contours dans le masque combiné
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialiser un compteur
n_red = 0
n_orange = 0

# Taille minimale d'une zone rouge ou orange
min_area = 2

# Analyser les contours détectés
for contour in contours:
    # Calculer la superficie de la zone entourée par le contour
    area = cv2.contourArea(contour)

    # Si la superficie est supérieure à la taille minimale, considérez cette zone comme rouge ou orange
    if area >= min_area:
        if cv2.contourArea(contour) >= min_area:
            n_red += 1
        else:
            n_orange += 1

# Afficher le nombre de zones rouges et oranges détectées
print("Nombre de zones rouges détectées :", n_red)
print("Nombre de zones oranges détectées :", n_orange)
