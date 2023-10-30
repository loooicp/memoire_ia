import cv2
import numpy as np

# Charger l'image
image = cv2.imread('1.jpg')

# Créer une copie de l'image pour l'affichage des rectangles
image_with_rectangles = image.copy()

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

# Taille minimale d'une zone rouge
min_area = 5 * 5

# Analyser les contours détectés
for contour in contours:
    # Calculer la superficie de la zone entourée par le contour
    area = cv2.contourArea(contour)

    # Si la superficie est supérieure à la taille minimale, considérez cette zone comme une zone rouge
    if area >= min_area:
        # Obtenir les coordonnées du rectangle englobant
        x, y, w, h = cv2.boundingRect(contour)
        
        # Dessiner un rectangle noir autour de la zone rouge
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 0, 0), 2)

# Enregistrer l'image avec les rectangles dessinés
cv2.imwrite('image_avec_rectangles.jpg', image_with_rectangles)

# Afficher l'image avec les rectangles (attendez que l'image s'affiche et appuyez sur une touche pour quitter)
cv2.imshow('Image avec rectangles', image_with_rectangles)
cv2.waitKey(0)
cv2.destroyAllWindows()
